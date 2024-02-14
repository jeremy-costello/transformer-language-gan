from torch import nn
from transformers import LlamaModel, LlamaForCausalLM


class TransformerGenerator(nn.Module):
    def __init__(self, config):
        self.transformer = LlamaForCausalLM(config)
    
    def forward(self, x):
        return self.transformer(x)

    def generate(self, inputs, *args, **kwargs):
        return self.transformer.generate(inputs, *args, **kwargs)


class TransformerDiscriminator(nn.Module):
    def __init__(self, config):
        self.transformer = LlamaModel(config)
        self.linear = nn.Linear(config.hidden_size, 1)
        # self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.linear(x).squeeze(-1)
        # x = self.activation(x)
        return x
