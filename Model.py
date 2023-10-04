import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder(nn.Module):
    def __init__(self, config):
        super(Embedder, self).__init__()

        self.token_embedder = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedder = nn.Embedding(config.sequence_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, idx):
        batch_size_sequence_length = idx.size() if idx.dim() == 2 else (1,idx.size()[0])
        batch_size = batch_size_sequence_length[0]
        sequence_length = batch_size_sequence_length[1]

        pos = torch.arange(0, sequence_length, dtype=torch.long, device=idx.device).unsqueeze(0).expand(batch_size, -1)

        token_embeddings = self.token_embedder(idx)
        positional_embeddings = self.position_embedder(pos)

        out = self.dropout(token_embeddings + positional_embeddings)

        return out
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadSelfAttention, self).__init__()

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.qkv_bias = config.qkv_bias
        self.head_dim = self.embed_dim // self.num_heads

        self.query = nn.Linear(self.embed_dim, self.embed_dim, bias=self.qkv_bias)
        self.key = nn.Linear(self.embed_dim, self.embed_dim, bias=self.qkv_bias)
        self.value = nn.Linear(self.embed_dim, self.embed_dim, bias=self.qkv_bias)

        self.fc = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        #Calculation of q,k,v
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim)

        #Reshape for Attention Calculation
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        #Calculate Scaled dot-product Attention
        attention = torch.matmul(q, k)
        attention = F.softmax(attention, dim=-1)
        attention = torch.matmul(attention, v)

        #Reshape/Flatten for FFNN 
        out = attention.permute(0,2,1,3).contiguous().view(batch_size, sequence_length, -1)
        out = self.fc(out)

        return out
    
class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNeuralNetwork, self).__init__()

        self.expand = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=config.ffnn_bias)
        self.gelu = nn.GELU()
        self.project = nn.Linear(4 * config.embed_dim, config.embed_dim, bias=config.ffnn_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        x = self.expand(x)
        x = self.gelu(x)
        x = self.project(x)
        x = self.dropout(x)
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, config):
        super(LayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(config.embed_dim))
        self.bias = nn.Parameter(torch.zeros(config.embed_dim)) if config.layernorm_bias else None

    def forward(self, x):
        out = F.layer_norm(x, self.weight.shape, self.weight, self.bias)
        return out
    
class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.MHSA = MultiHeadSelfAttention(config)
        self.FFNN = FeedForwardNeuralNetwork(config)
        self.LN1 = LayerNorm(config)
        self.LN2 = LayerNorm(config)

    def forward(self, x):
        x = x + self.MHSA(x)
        x = self.LN1(x)
        x = x + self.FFNN(x)
        x = self.LN2(x)
        return x
    
class PostTransformerLayers(nn.Module):
    def __init__(self, config):
        super(PostTransformerLayers, self).__init__()

        self.fc = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
    
    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x
    
def PostProcess(probs, topK, temperature):
    sorted_probs, sorted_indices = torch.topk(probs, topK)
    flattened_probs, flattened_indices = sorted_probs.view(-1), sorted_indices.view(-1)
    next_token = torch.multinomial(torch.softmax(flattened_probs / temperature, dim=0), 1)
    next_token = flattened_indices[next_token]
    return next_token

class PashkoModel(nn.Module):
    def __init__(self, config):
        super(PashkoModel, self).__init__()

        self.Encoder = tiktoken.get_encoding('gpt2')
        self.Embedder = Embedder(config)
        
        self.Blocks = [Block(config) for _ in range(config.num_blocks)]
        for i, block in enumerate(self.Blocks):
            self.add_module(f'Transformer Block {i}', block)

        self.PostTransformer = PostTransformerLayers(config)

        self.LossFunction = nn.CrossEntropyLoss()

        self.config = config

        #Weight Initialisation according to GPT2 Paper
        self.apply(self.init_weights)

        #Weight tying Embedding to Final Linear
        self.Embedder.token_embedder.weight = self.PostTransformer.fc.weight

    #Different types of generation.
    def forward(self, x, targets=None):
        x = self.Embedder(x) #Embeddings

        for Transformer in self.Blocks: #Transformer Block
            x = Transformer(x)
        
        x = self.PostTransformer(x) #Post-Transformer Layers

        x = x.view(-1, self.config.vocab_size) #Loss Calculation
        targets = targets.view(-1)

        padding_mask = (targets != -1)
        loss = self.LossFunction(x[padding_mask], targets[padding_mask])
        return x, loss
    
    @torch.no_grad()
    def inference(self, x):
        x = self.Embedder(x) #Embeddings

        for Transformer in self.Blocks: #Transformer Block
            x = Transformer(x)
        
        x = self.PostTransformer(x) #Post-Transformer Layers
        token = PostProcess(x, self.config.topK, self.config.temperature)
        return token
    
    @torch.no_grad()
    def generate(self, context, max_new_tokens=64, show=True):
        response = []

        x = torch.LongTensor(self.Encoder.encode(context))
        next_token = 0

        for _ in range(max_new_tokens):
            x = x if x.size(0) <= self.config.sequence_length else x[:, -self.config.sequence_length:]

            next_token = self.inference(x)

            if next_token == 50256:
                return response

            next_token = '' if next_token == -1 else next_token #Exclude padding tokens

            response.append(next_token.numpy()[0])

            if show:
                print(self.Encoder.decode(next_token.numpy()), end='', flush=True)

            x = torch.cat((x, next_token), dim=0)
        
        return response
    
    #Utils
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, embedding=False):
        num_params = sum(p.numel() for p in self.parameters())
        if not embedding:
            num_params -= self.Embedder.position_embedder.weight.numel()
        return "{:.2f}M".format(num_params / 1000000), num_params
    
def move_modules_to_device(module, device):
    for child in module.children():
        if isinstance(child, torch.nn.Module):
            move_modules_to_device(child, device)
    module.to(device)