import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical

from configs import get_configs
from data_processing import preprocess_text, TextDataset
from models import TransformerGenerator, TransformerDiscriminator


device = "cuda"
num_epochs = 10
batch_size = 32
num_workers = 3
context_length = 128
text_file = "./text/input.txt"

vocab_size, tokenizer, token_probs, full_text_array = preprocess_text(text_file)

generator_config, discriminator_config = get_configs(vocab_size, context_length)

generator = TransformerGenerator(generator_config).to(device)
discriminator = TransformerDiscriminator(discriminator_config).to(device)

# add weight decay
generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=3e-4)
discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=3e-4)

for epoch in range(num_epochs):
    train_dataset = TextDataset(full_text_array, context_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    for batch in train_loader:
        real_inputs = batch.to(device)
        real_logits = discriminator(real_inputs)
        real_labels = torch.ones(batch_size, context_length, device=device)
        
        z = torch.multinomial(token_probs, num_samples=batch_size).view(-1, 1)
        fake_inputs = generator.generate(z, max_length=context_length, temperature=1.0, top_p=1.0, do_sample=True, output_scores=True, return_dict_in_generate=True)
        fake_logits = discriminator(fake_inputs.sequences)
        fake_labels = torch.zeros(batch_size, context_length, device=device)
        
        discriminator_optimizer.zero_grad()
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
        real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
        discriminator_loss = 0.5 * (fake_loss + real_loss)
        discriminator_loss.backward()
        # clipping
        discriminator_optimizer.step()
        
        generator_optimizer.zero_grad()
        fake_predictions = F.sigmoid(fake_logits)
        R = 2 * fake_predictions - 1
        # https://github.com/urchade/molgen/blob/master/layers.py
        # probably need to write custom generation code
        for logits in fake_inputs.scores:
            dist = Categorical(logits=logits)
            # WIP
