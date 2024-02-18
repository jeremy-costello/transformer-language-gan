from transformers import LlamaConfig


def get_configs(vocab_size, context_length):
    generator_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        max_position_embeddings=context_length,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0
    )

    discriminator_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        max_position_embeddings=context_length,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        num_labels=1
    )
    
    return generator_config, discriminator_config
