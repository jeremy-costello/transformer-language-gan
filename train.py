import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import get_configs
from data_processing import preprocess_text, TextDataset
from models import TransformerGenerator, TransformerDiscriminator


device = "cuda"
num_epochs = 1000
batch_size = 128
num_workers = 3
context_length = 32
input_text_file = "./text/test.txt"

entropy_mult = 0.0
moving_average_period = 10
clip_value = 1.0
output_text_file_path = "outputs.txt"
gamma = 0.9

steps_per_epoch = 40
generator_lr = 1e-4
discriminator_lr = 1e-2

real_label_target = 0.9
discriminator_accumulation_steps = 2
discriminator_loss_function = F.binary_cross_entropy_with_logits
schedule_learning_rate = False

# generator
beta1 = 0.5
beta2 = 0.95


vocab_size, tokenizer, full_text_array = preprocess_text(input_text_file)

generator_config, discriminator_config = get_configs(vocab_size, context_length)

generator = TransformerGenerator(generator_config).to(device)
discriminator = TransformerDiscriminator(discriminator_config, context_length=context_length).to(device)

# add weight decay and no weight decay groups
# add LR schedulers
generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=generator_lr, betas=(beta1, beta2))
if schedule_learning_rate:
    gen_opt_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=generator_optimizer,
        max_lr=generator_lr,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        base_momentum=0.4,
        max_momentum=0.9
    )

discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=discriminator_lr)
if schedule_learning_rate:
    disc_opt_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=discriminator_optimizer,
        max_lr=discriminator_lr,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        base_momentum=0.4,
        max_momentum=0.9
    )

baseline = 0

if os.path.isfile(output_text_file_path):
    os.remove(output_text_file_path)
    
with open(output_text_file_path, "a") as output_text_file:
    average_discriminator_loss = -1
    cumulative_discriminator_loss = 0
    
    for epoch in tqdm(range(num_epochs)):
        train_dataset = TextDataset(full_text_array, context_length, batch_size, discriminator_accumulation_steps)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        for iter, batch in enumerate(tqdm(train_loader, leave=False)):
            index = iter % context_length
            if index == 0:
                fake_generations = torch.zeros((batch_size, context_length), dtype=torch.int64, device=device)
                rewards = []
                log_probs = []
                entropies = []
            
            is_accumulating = True
            if (iter + 1) % context_length == 0:
                is_accumulating = False
            
            discriminator_is_accumulating = True
            if (iter + 1) % (discriminator_accumulation_steps * context_length) == 0:
                discriminator_is_accumulating = False  # should be false
                
            real_inputs = batch.to(device)
            real_logits = discriminator(real_inputs, index=index)
            real_labels = real_label_target * torch.ones((batch_size, 1), device=device)
            
            if index == 0:
                input_ids = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
                past_key_values = None
            else:
                input_ids = fake_generations.clone().detach()[:, :index]  # clone, detach?
                past_key_values = None
                
            fake_inputs = generator.generate(
                input_ids=input_ids,
                past_key_values=past_key_values,
                temperature=1.0,  # could possibly add temperature as part of the action space
                top_p=1.0
            )
            fake_generations[:, index] = fake_inputs["output_id"].squeeze(-1)
            
            current_fake_generation = torch.cat(
                (
                    torch.zeros((batch_size, context_length - index - 1), dtype=torch.int64, device=device),
                    fake_generations.clone().detach()[:, :index + 1]
                ),
                dim=1
            )
            
            fake_logits = discriminator(current_fake_generation, index=index)
            fake_labels = torch.zeros((batch_size, 1), device=device)
            
            fake_loss = discriminator_loss_function(fake_logits, fake_labels)
            real_loss = discriminator_loss_function(real_logits, real_labels)
            discriminator_loss = 0.5 * (fake_loss + real_loss)
            discriminator_loss.backward()
            
            cumulative_discriminator_loss += discriminator_loss.item()
            
            if not discriminator_is_accumulating:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_value)
                discriminator_optimizer.step()
                if schedule_learning_rate:
                    disc_opt_scheduler.step()
                discriminator_optimizer.zero_grad()
                
                average_discriminator_loss = cumulative_discriminator_loss / (discriminator_accumulation_steps * context_length)
                cumulative_discriminator_loss = 0
            
            fake_predictions = F.sigmoid(fake_logits)
            
            reward = 2.0 * fake_predictions - 1.0
            
            rewards.append(reward.squeeze(-1))
            log_probs.append(fake_inputs["log_prob"])
            entropies.append(fake_inputs["entropy"])
            
            if not is_accumulating:
                # https://github.com/google-deepmind/deepmind-research/blob/master/scratchgan/losses.py#L67
                cumulative_rewards = []
                for t in range(context_length):
                    cum_value = torch.zeros((batch_size,), device=device)
                    for s in range(t, context_length):
                        cum_value += np.power(gamma, (s - t)) * rewards[s]
                    cumulative_rewards.append(cum_value)
                
                rewards = torch.stack(rewards, dim=0)
                cumulative_rewards = torch.stack(cumulative_rewards, dim=0)
                
                mean_reward = torch.mean(rewards)
                mean_cumulative_reward = torch.mean(cumulative_rewards)
                baseline = (1.0 - 1.0 / moving_average_period) * baseline \
                    + (1.0 / moving_average_period) * mean_cumulative_reward
                
                cumulative_generator_loss = 0
                for cumulative_reward, log_prob, entropy in zip(cumulative_rewards, log_probs, entropies):
                    advantage = cumulative_reward - baseline
                    log_loss = -1.0 * torch.mean(advantage.detach() * log_prob)
                    entropy_loss = -1.0 * entropy_mult * torch.mean(entropy)
                    generator_loss = log_loss + entropy_loss
                    generator_loss.backward()
                    
                    cumulative_generator_loss += generator_loss.item()

                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_value)
                generator_optimizer.step()
                if schedule_learning_rate:
                    gen_opt_scheduler.step()
                generator_optimizer.zero_grad()
                
                average_generator_loss = cumulative_generator_loss / context_length
            
                print(
                    "\n"
                    f"discriminator loss: {round(average_discriminator_loss, 4)}, "
                    f"generator loss: {round(average_generator_loss, 4)}, "
                    f"average reward: {round(mean_reward.item(), 4)}"
                    "\n"
                )
            
                example_index = torch.randint(batch_size, size=(1,)).item()
                example_list = fake_generations[example_index].detach().cpu().tolist()
                example_string = "".join([tokenizer["idx2token"][idx] for idx in example_list])
                output_text_file.write(example_string + "\n\n")
