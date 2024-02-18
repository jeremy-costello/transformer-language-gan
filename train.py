import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import get_configs
from data_processing import preprocess_text, TextDataset
from models import TransformerGenerator, TransformerDiscriminator


device = "cuda"
num_epochs = 1000
batch_size = 1024
num_workers = 3
context_length = 128
input_text_file = "./text/input.txt"

entropy_mult = 0.01
moving_average_period = 10
clip_value = 1.0
output_text_file_path = "outputs.txt"
learning_rate = 3e-4


vocab_size, tokenizer, token_probs, full_text_array = preprocess_text(input_text_file)

generator_config, discriminator_config = get_configs(vocab_size, context_length)

generator = TransformerGenerator(generator_config).to(device)
discriminator = TransformerDiscriminator(discriminator_config).to(device)

# add weight decay and no weight decay groups
generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate)
discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=learning_rate)

baseline = 0

with open(output_text_file_path, "a") as output_text_file:
    for epoch in tqdm(range(num_epochs)):
        train_dataset = TextDataset(full_text_array, context_length, batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        for iter, batch in enumerate(tqdm(train_loader, leave=False)):
            index = iter % context_length
            if index == 0:
                fake_generations = torch.zeros((batch_size, context_length), dtype=torch.int64, device=device)
                cumulative_reward = 0
            
            is_accumulating = True
            if (iter + 1) % context_length == 0:
                is_accumulating = False
                
            real_inputs = batch.to(device)
            real_logits = discriminator(real_inputs)
            real_labels = torch.ones((batch_size, 1), device=device)
            
            if index == 0:
                input_id = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
                past_key_values = None
            else:
                input_id = fake_inputs["output_id"]
                past_key_values = fake_inputs["past_key_values"]                
            
            fake_inputs = generator.generate(
                input_id=input_id,
                past_key_values=past_key_values,
                temperature=1.0  # could possibly add temperature as part of the action space
            )
            fake_generations[:, index] = fake_inputs["output_id"].squeeze(-1)
            
            fake_logits = discriminator(fake_generations)
            fake_labels = torch.zeros((batch_size, 1), device=device)
            
            fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
            real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
            discriminator_loss = 0.5 * (fake_loss + real_loss)
            discriminator_loss.backward()
            
            if not is_accumulating:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_value)
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad()
            
            fake_predictions = F.sigmoid(fake_logits.detach())
            
            # add masking to reward and attention mask to discriminator
            reward = 2 * fake_predictions - 1
            
            log_loss = -1.0 * torch.mean((reward - baseline) * fake_inputs["log_prob"])
            entropy_loss = -1.0 * entropy_mult * torch.mean(fake_inputs["entropy"])
            generator_loss = log_loss + entropy_loss
            
            with torch.no_grad():
                cumulative_reward += torch.mean(reward)
            
            # why does this need retain_graph ???
            generator_loss.backward(retain_graph=True)
            
            if not is_accumulating:
                with torch.no_grad():
                    mean_reward = cumulative_reward / context_length
                    baseline = (1.0 - 1.0 / moving_average_period) * baseline \
                        + (1.0 / moving_average_period) * mean_reward

                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_value)
                generator_optimizer.step()
                generator_optimizer.zero_grad()
            
                print(
                    "\n"
                    f"discriminator loss: {round(discriminator_loss.item(), 4)}, "
                    f"generator loss: {round(generator_loss.item(), 4)}, "
                    f"average reward: {round(mean_reward.item(), 4)}"
                )
            
                example_index = torch.randint(batch_size, size=(1,)).item()
                example_list = fake_generations[example_index].detach().cpu().tolist()
                example_string = "".join([tokenizer["idx2token"][idx] for idx in example_list])
                output_text_file.write(example_string + "\n\n")
