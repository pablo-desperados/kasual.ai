"""Input embeddings: token IDs -> the vectors the transformer layers consume.
"""
from torch import nn


class KasualInputEmbedding(nn.Module):
    """(batch, seq_len) token IDs -> (batch, seq_len, emb_dim) vectors.

    Args:
        vocab_size: rows in the token table; use len(tokenizer) so it always
            matches the configured tokenizer (163,840 for Kimi K3).
        emb_dim: length of each vector (the model's width).
    """

    def __init__(self, vocab_size, emb_dim, context_length=None):
        super().__init__()
        
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        
        self.context_length = context_length
    
    def forward(self, token_ids):
        lookup = self.tok_emb(token_ids)
        return lookup
