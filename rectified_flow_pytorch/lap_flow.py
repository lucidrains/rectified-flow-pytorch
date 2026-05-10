from huggingface_hub.inference._generated.types import zero_shot_image_classification
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
import math

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)
    

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
      
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

class AdaLNModulation(Module):
    def __init__(self, dim, out_multiplier=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, out_multiplier * dim, bias=True)
        )
       
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, c):
        return self.net(c)

class DiTTransformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, patch_dim, dropout = 0.):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6),
                FeedForward(dim, mlp_dim, dropout = dropout),
                AdaLNModulation(dim)
            ]))

        self.adaln_final = AdaLNModulation(dim, out_multiplier=2)
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear_final = nn.Linear(dim, patch_dim, bias=True)

        nn.init.constant_(self.linear_final.weight, 0)
        nn.init.constant_(self.linear_final.bias, 0)
        
    
    def forward(self, x, c):
        for norm1, attn, norm2, ff, adaln in self.layers:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln(c).chunk(6, dim=-1)
            
            x = x + gate_msa.unsqueeze(1) * attn(modulate(norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * ff(modulate(norm2(x), shift_mlp, scale_mlp))

        shift_final, scale_final = self.adaln_final(c).chunk(2, dim=-1)

        x = modulate(self.norm_final(x), shift_final, scale_final)

        x = self.linear_final(x)

        return x



class DiT(Module):
    def __init__(
        self, 
        image_size, 
        patch_size, 
        num_classes, 
        dim, 
        depth, 
        heads, 
        mlp_dim, 
        channels = 3, 
        dim_head = 64, 
        dropout = 0.
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim))
        self.dropout = nn.Dropout(dropout)
        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = nn.Embedding(num_classes, dim)

        self.transformer = DiTTransformer(dim, depth, heads, dim_head, mlp_dim, patch_dim, dropout)

        self.unpatchify = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                                    h = image_height // patch_height, 
                                    w = image_width // patch_width, 
                                    p1 = patch_height, p2 = patch_width)

        nn.init.normal_(self.y_embedder.weight, std=0.02)

    def forward(self, img, t, y):
        x = self.to_patch_embedding(img)
        x = x + self.pos_embedding
        x = self.dropout(x)

        c = self.t_embedder(t) + self.y_embedder(y)

        x = self.transformer(x, c)
        
        return self.unpatchify(x)

if __name__ == "__main__":
    batch_size = 2
    image_size = 32
    patch_size = 4
    num_classes = 10
    dim = 128
    depth = 4
    heads = 4
    mlp_dim = 256
    
    model = DiT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim
    )
    
    img = torch.randn(batch_size, 3, image_size, image_size)
    t = torch.randn(batch_size)
    y = torch.randint(0, num_classes, (batch_size,))
    
    output = model(img, t, y)
    
    print(f"Input shape: {img.shape}")
    print(f"Output shape: {output.shape}")