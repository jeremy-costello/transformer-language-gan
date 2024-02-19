from transformers import LlamaConfig


def get_configs(vocab_size, context_length):
    generator_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=context_length,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        rope_theta=500.0,
        tie_word_embeddings=True
    )

    discriminator_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=context_length,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        rope_theta=500.0,
        attention_dropout=0.5,
        num_labels=1
    )
    
    return generator_config, discriminator_config
