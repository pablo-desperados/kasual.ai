# Multihead Attention Class for Kasual Ai Model and training

from pre_process.kasualconfig import KasualConfig
import torch
from torch import nn ,Tensor


class KasualAttention(nn.Module):
    def __init__(self, config: KasualConfig):
        super().__init__()
        self.num_heads = config.attention.num_heads
        self.head_dim = config.head_dims_split
        self.attention_dropout = config.attention.attention_dropout
        self.embedding_dim = config.preprocessor.embedding_dim
        self.attention_mechanism = config.attention.implementation
        
        # Projections for query, key, and value. Needed for attention weights calculation and context vector generation
        self.query_matrix = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.key_matrix = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.value_matrix = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.dropout = nn.Dropout(self.attention_dropout)
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.scaling = self.head_dim ** -0.5  # Scaling factor for attention scores
        
    def forward(self, x: Tensor, document_ids: Tensor | None = None, return_weight=False):
        b, t, _ = x.shape
        
        query = self.query_matrix(x)
        keys = self.key_matrix(x)
        values = self.value_matrix(x)
        
       # split operation for multi-head attention
       
        split_query = self.split_heads(query)
        split_keys = self.split_heads(keys)
        split_values = self.split_heads(values)
        
        if self.attention_mechanism == 'sdpa' and not return_weight:
            context_vectors = nn.functional.scaled_dot_product_attention(
                split_query, split_keys, split_values,
                attn_mask= None if document_ids is None else ~self.mask_build(t, document_ids, x.device),
                is_causal= document_ids is None,
                dropout_p= self.dropout.p if self.training else 0.0,
                scale= self.scaling,
            )
            attention_weights=None
        else:
            #Calc attention scores by taking dot for query and key
            attn_scores = split_query @ split_keys.transpose(-2, -1) * self.scaling
            
            #Masking matrix to prevent attention to future tokens
            mask = self.mask_build(t, document_ids, x.device)
                
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
            
            #Softmax to get attention weights
            attention_weights = torch.softmax(attn_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            #Context vectors
            context_vectors = attention_weights @ split_values
        
        context = context_vectors.transpose(1, 2).contiguous().view(b, t, self.embedding_dim) 
        
        output = self.out_proj(context)
        return (output, attention_weights) if return_weight else output
        

        
    def mask_build(self, tokens, document_ids, device):
        mask: Tensor = torch.triu(torch.ones(tokens,tokens, dtype=torch.bool, device=device), diagonal=1)
        if document_ids is not None:
            same_doc = document_ids[:, :, None] == document_ids[:, None, :]
            # get OR op to only reveal tokens from the current document
            mask = mask | ~same_doc
            mask = mask.unsqueeze(1)
        return mask
        
    def split_heads(self, x: Tensor) -> Tensor:
        b, t, d = x.size()
        x = x.view(b, t, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (b, num_heads, t, head_dim) 
        
        
        
