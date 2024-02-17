import torch
from torch import nn
from torch.distributions import Categorical
from transformers import LlamaForCausalLM, LlamaForSequenceClassification


class TransformerGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = LlamaForCausalLM(config)
    
    def forward(self, input_ids, past_key_values=None):
        return self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values
        )

    # custom generation function
    def generate(self, device, batch_size, length, temperature=1.0, top_p=1.0, fill_value=-float("inf")):
        input_id = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
        past_key_values = None
        
        generations = torch.empty((batch_size, length), dtype=torch.int64, device=device)
        log_probs = torch.empty((batch_size, length), dtype=torch.float32, device=device)
        entropies = torch.empty((batch_size, length), dtype=torch.float32, device=device)
        
        for index in range(length):
            output = self(
                input_ids=input_id,
                past_key_values=past_key_values
            )
            logits = output.logits
            past_key_values = output.past_key_values
            
            # temperature
            logits = logits / temperature
            
            # top-p sampling. doesn't work with categorical
            if top_p < 1.0:
                # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/generation/logits_process.py#L447
                sorted_logits, sorted_indices = torch.sort(logits, descending=False)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                
                # remove tokens with cumulative top_p above the threshold (tokens with 0 are kept)
                sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
                
                # scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, fill_value)
            
            # categorical distribution
            dist = Categorical(logits=logits)
            
            input_id = dist.sample()
            log_prob = dist.log_prob(input_id)
            entropy = dist.entropy()
            
            generations[:, index] = input_id.squeeze(-1)
            log_probs[:, index] = log_prob.squeeze(-1)
            entropies[:, index] = entropy.squeeze(-1)
        
        return {
            "generations": generations,
            "log_probs": log_probs,
            "entropies": entropies
        }


class TransformerDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        # other option is LlamaModel and nn.Linear from 64 -> 1
        self.transformer = LlamaForSequenceClassification(config)
    
    def forward(self, x):
        x = self.transformer(x)
        return x.logits
