from typing import Tuple, Union, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import numpy as np
from einops import rearrange, reduce

from models.clip.model import CLIP, LayerNorm, Transformer, QuickGELU
import models.clip as clip
from models.visual_transformers import resize_pos_embed, inflate_weight

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    print("Install Fairscale >= 0.3.7 first to enable gradient checkpoint")
    checkpoint_wrapper = None

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    flash_attn_unpadded_func = None


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        # assert flash_attn_unpadded_func is not None, "FlashAttention is not installed."
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
                        kdim, vdim, batch_first, device, dtype)

    def attention(
        self,
        q, k, v,
        batch_size=1,
        seqlen=77,
        softmax_scale=None,
        attention_dropout=0.0,
        causal=False,
        cu_seqlens=None,
        max_s=None,
        need_weights=False
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q,k,v: The tensor containing the query, key, and value. each of (B*S, H, D)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda

        if cu_seqlens is None:
            max_s = seqlen
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                    device=q.device)
            output = flash_attn_unpadded_func(
                q, k, v, cu_seqlens, cu_seqlens, max_s, max_s, attention_dropout,
                softmax_scale=softmax_scale, causal=causal
            )

        return output

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # set up shape vars
        seqlen, batch_size, embed_dim = query.shape
        
        # in-projection and rearrange `s b (3 h d) -> s b (h d) -> (b s) h d`
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.transpose(0, 1).contiguous().view(batch_size * seqlen, self.num_heads, self.head_dim)
        k = k.transpose(0, 1).contiguous().view(batch_size * seqlen, self.num_heads, self.head_dim)
        v = v.transpose(0, 1).contiguous().view(batch_size * seqlen, self.num_heads, self.head_dim)
        
        # flash attention (use causal mask)
        causal = attn_mask is not None
        attn_output = self.attention(q, k, v, batch_size, seqlen, causal=causal)

        # out-projection
        # `(b s) h d -> s b (h d)`
        attn_output = attn_output.contiguous().view(batch_size, seqlen, self.num_heads, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(seqlen, batch_size, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, None


class Local_MHRA(nn.Module):
    def __init__(self, d_model, dw_reduction=1.5, pos_kernel_size=3):
        super().__init__() 

        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)
        self.ln = LayerNorm(d_model)
        self.pos_embed = nn.Sequential(
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(re_d_model, re_d_model, kernel_size=(pos_kernel_size, 1, 1), stride=(1, 1, 1), padding=(padding, 0, 0), groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )

        # init zero
        print('Init zero for Conv in pos_emb')
        nn.init.constant_(self.pos_embed[2].weight, 0)
        nn.init.constant_(self.pos_embed[2].bias, 0)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pos_embed(x)
        return x


class TemporalBlock(nn.Module):
    def __init__(
            self, d_model, n_head, attn_mask=None, drop_path=0.0, 
            dw_reduction=1.5, config=None
        ):
        super().__init__() 
        
        self.n_head = n_head
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.config = config
        self.lmhra1 = Local_MHRA(d_model, dw_reduction=dw_reduction)
        if self.config.get('double_lmhra', False):
            self.lmhra2 = Local_MHRA(d_model, dw_reduction=dw_reduction)

        # spatial
        if flash_attn_unpadded_func:
            print("Using Flash Attention")
            self.attn = MultiheadAttention(d_model, n_head)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x, T=8, use_checkpoint=False):
        # x: 1+HW, NT, C
        tmp_x = x[1:, :, :]
        L, NT, C = tmp_x.shape
        N = NT // T
        H = W = int(L ** 0.5)
        tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()
        tmp_x = tmp_x + self.drop_path(self.lmhra1(tmp_x))
        tmp_x = tmp_x.view(N, C, T, L).permute(3, 0, 2, 1).contiguous().view(L, NT, C)
        x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        # MHSA
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        # Local MHRA
        if self.config.get('double_lmhra', False):
            tmp_x = x[1:, :, :]
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra2(tmp_x))
            tmp_x = tmp_x.view(N, C, T, L).permute(3, 0, 2, 1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        # FFN
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None, use_checkpoint=False, T=8, config=None):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)] 
        self.width = width
        self.layers = layers
        if use_checkpoint and checkpoint_wrapper:
            self.resblocks = nn.ModuleList([
                checkpoint_wrapper(TemporalBlock(width, heads, attn_mask, droppath[i], config=config))
                for i in range(layers)])
        else:
            self.resblocks = nn.ModuleList([TemporalBlock(width, heads, attn_mask, droppath[i], config=config) 
                for i in range(layers)])
       
    def forward(self, x: torch.Tensor, T):
        L, NT, C = x.shape
        N = NT // T
        H = W = int((L - 1) ** 0.5)
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, T)
        return x


class VideoFormer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 droppath = None, T = 8, use_checkpoint = False, config=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.temporal_stride = config.get('temporal_stride', 2)
        self.temporal_downsampling = config.get('temporal_downsampling', False)
        
        if checkpoint_wrapper and use_checkpoint:
            self.conv1 = checkpoint_wrapper(nn.Conv3d(
                in_channels=3, out_channels=width, 
                kernel_size=(3, patch_size, patch_size) if self.temporal_downsampling else (1, patch_size, patch_size),
                stride=(self.temporal_stride, patch_size, patch_size) if self.temporal_downsampling else (1, patch_size, patch_size),
                padding=(1, 0, 0) if self.temporal_downsampling else (0, 0, 0),
                bias=False
            ))
        else:
            self.conv1 = nn.Conv3d(
                in_channels=3, out_channels=width, 
                kernel_size=(3, patch_size, patch_size) if self.temporal_downsampling else (1, patch_size, patch_size),
                stride=(self.temporal_stride, patch_size, patch_size) if self.temporal_downsampling else (1, patch_size, patch_size),
                padding=(1, 0, 0) if self.temporal_downsampling else (0, 0, 0),
                bias=False
            )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        ## Attention Blocks
        self.transformer = TemporalTransformer(width, layers, heads, droppath=droppath, use_checkpoint=use_checkpoint, config=config)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        ## Temporal Related
        self.num_frames = T
        self.config = config
        
        # Initalization
        trunc_normal_(self.positional_embedding, std=.02)
        trunc_normal_(self.class_embedding, std=.02)
        # self.apply(self._init_weights)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, skip_last_layer=False, use_checkpoint=True):
        B, C, T, H, W = x.shape
        x = self.conv1(x)
        T, H, W = x.shape[-3:]
        x = rearrange(x, 'b d t h w -> (b t) (h w) d')
       
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x, T)
        # x  1+HW, NT, C

        # x = rearrange(x, 'n (b t) d -> b t n d', b=B)
        # cls_token = x[:,:,0,:].mean(1).unsqueeze(1) # [b, 1, d]
        # x = x[:,:,1:,:]
        # x = rearrange(x, 'b t n d -> b (t n) d')
        # x = torch.cat([cls_token, x], dim=1)
        x = rearrange(x, 'n bt d -> bt n d')
        x = self.ln_post(x)

        return x


class XCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int, 
                 # video
                 T=8, 
                 droppath=0.,
                 # other
                 use_cache=False,
                 use_checkpoint=True,
                 config=None
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        
        # self.prompts_generator = VideoSpecificPrompt(layers=prompts_layers, embed_dim=vision_width, alpha=prompts_alpha,)
        self.use_cache=use_cache

        dpr = [x.item() for x in torch.linspace(0, droppath, vision_layers)] if droppath > 0. else None

        vision_heads = vision_width // 64
        # if config.get('videoformer', True):
        if True:
            self.visual = VideoFormer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                droppath=dpr,
                T=T,
                use_checkpoint=use_checkpoint,
                config=config
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cache_text_features = None
        
        self.initialize_parameters()
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def encode_image(self, image, skip_last_layer=False):
        return self.visual(image, skip_last_layer)

    def encode_text(self, text):
        x = self.token_embedding(text)
        eos_indx = text.argmax(dim=-1)
        K, N1, C = x.shape

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(K, -1)
        return x


def build_model(state_dict: dict, img_size=256, T=8, droppath=0., use_checkpoint=False, use_cache=True, config=None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = XCLIP(
        embed_dim,
        img_size, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,  
        T=T, droppath=droppath, use_checkpoint=use_checkpoint, use_cache=use_cache, config=config
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    
    patch_video = model.state_dict()['visual.conv1.weight']
    kernel_t = patch_video.shape[2]
    new_weight = inflate_weight(state_dict['visual.conv1.weight'], kernel_t, center=True)
    state_dict['visual.conv1.weight'] = new_weight

    num_patches = int(img_size*img_size/(vision_patch_size*vision_patch_size))
    pos_embed = nn.Parameter(torch.zeros(num_patches + 1, vision_width).float())    
    pos_embed.weight = resize_pos_embed(state_dict['visual.positional_embedding'].unsqueeze(0), pos_embed.unsqueeze(0))
    state_dict['visual.positional_embedding'] = pos_embed

    msg = model.load_state_dict(state_dict,strict=False)
    print(msg)
    return model.eval()


def load(model_path, name: str="", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", 
         jit=True, T=8, droppath=0., use_checkpoint=False, use_cache=True, img_size=256, config=None
):
    if model_path is None:
        model_path = clip._download(clip._MODELS[name])
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), img_size=img_size, T=T, droppath=droppath, 
                        use_checkpoint=use_checkpoint,
                        use_cache=use_cache,
                        config=config
                        )
    if str(device) == "cpu":
        model.float()
    return model, model.state_dict()