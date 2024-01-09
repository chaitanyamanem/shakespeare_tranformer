import torch
from torch import nn
import numpy as np



class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, vocab_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril = torch.tril(torch.ones(block_size, block_size)).to("cuda")   
        self.dropout = nn.Dropout(0.2)     
        
    def forward(self,X,y=None):
        q = self.query(X)
        k = self.key(X)
        v = self.value(X)
        wei = q @ k.transpose(-2, -1) / np.sqrt(k.shape[-1])
        wei = wei.masked_fill(self.tril==0, float('-inf'))
        wei = torch.softmax(wei, axis=-1)
        wei = self.dropout(wei)
        out = wei @ v       
        
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embedding_dim, head_size, block_size, vocab_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(embedding_dim, head_size, block_size, vocab_size) for _ in range(n_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self,X,y=None):
        out = torch.cat([head(X) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForword(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.2)
        )
    def forward(self, X):
        return self.layer(X)
    
class Block(nn.Module):
    def __init__(self, n_heads, embedding_dim, head_size, block_size, vocab_size):
        super().__init__()
        self.mh = MultiHeadAttention(n_heads, embedding_dim, head_size, block_size, vocab_size)
        self.ff = FeedForword(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):                
        x = x + self.mh(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, head_size, n_heads, n_blocks):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential( *[Block(n_heads, embedding_dim, head_size, block_size, vocab_size) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(embedding_dim)        
        self.lm_head = nn.Linear(embedding_dim, vocab_size)      
        
        
    def forward(self,X,y=None):
        B, T = X.shape
        tokens = self.token_embedding(X)
        positions = self.position_embedding(torch.arange(T, device='cuda'))
        inputs = tokens + positions
        x = self.blocks(inputs)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
        
    def generate(self, prompt, max_new_tokens, block_size):

        for i in range(max_new_tokens):
            prompt = prompt[:-block_size]
            pred = self(prompt)
            pred = pred[:,-1,:]
            probs = torch.softmax(pred, axis=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            prompt = torch.cat([prompt, next_idx], axis=1)
        
        return prompt

        


    

