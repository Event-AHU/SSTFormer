import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
from timm.models import create_model
import pdb
from MST_model import MST
from Bottleneck import BottleneckTransformer
# from mmengine.model.weight_init import kaiming_init
from mmcv.cnn import kaiming_init

__all__ = ['Spikingformer']
tau_thr = 1.5

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True)  #, backend='cupy'
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True)
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        return x


class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.proj_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True)

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True)

        self.attn_lif = MultiStepLIFNode(tau=tau_thr, v_threshold=0.5, detach_reset=True)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)


    def forward(self, x):
        T, B, C, H, W = x.shape  #torch.Size([4, 2, 256, 21, 16])
        x = self.proj_lif(x)

        x = x.flatten(3)   #torch.Size([4, 2, 256, 336])
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)  #torch.Size([8, 256, 336])

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4) #torch.Size([4, 2, 16, 336, 16])


        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)#torch.Size([4, 2, 16, 336, 16])

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)  #torch.Size([4, 2, 16, 336, 16])

        attn = (q @ k.transpose(-2, -1))  #torch.Size([4, 2, 16, 336, 336])
        x = (attn @ v) * 0.125  #torch.Size([4, 2, 16, 336, 16])

        x = x.transpose(3, 4).reshape(T, B, C, N)
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)  #torch.Size([4, 2, 256, 21, 16])

        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SpikingTokenizer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)

        self.proj1_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_conv = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims//4)

        self.proj2_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(3,2), stride=2, padding=0, dilation=1, ceil_mode=False)
        self.proj2_conv = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims//2)

        self.proj3_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(2,3), stride=2, padding=0, dilation=1, ceil_mode=False)
        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)

        self.proj4_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=(3,2), stride=2, padding=0, dilation=1, ceil_mode=False)
        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, C, H, W = x.shape  #torch.Size([4, 2, 12, 346, 260])
        x = self.proj_conv(x.flatten(0, 1)) # have some fire value torch.Size([8, 32, 346, 260])
        x = self.proj_bn(x).reshape(T,B,-1,H,W).contiguous()  #torch.Size([4, 2, 32, 346, 260])

        x = self.proj1_lif(x).flatten(0,1).contiguous()  #torch.Size([8, 32, 346, 260])
        x = self.maxpool1(x)                             #torch.Size([8, 32, 173, 130])
        x = self.proj1_conv(x)                           #torch.Size([8, 64, 173, 130])
        x = self.proj1_bn(x).reshape(T, B, -1, H//2, W//2).contiguous() #torch.Size([4, 2, 64, 173, 130])

        x = self.proj2_lif(x).flatten(0, 1).contiguous() #torch.Size([8, 64, 173, 130])
        x = self.maxpool2(x)                               #torch.Size([8, 64, 86, 65])
        x = self.proj2_conv(x)                             #torch.Size([8, 128, 86, 65])
        x = self.proj2_bn(x).reshape(T, B, -1, H//4, W//4).contiguous() #torch.Size([4, 2, 128, 86, 65])

        x = self.proj3_lif(x).flatten(0, 1).contiguous()    #torch.Size([8, 128, 86, 65])
        x = self.maxpool3(x)                            #torch.Size([8, 128, 43, 32])
        x = self.proj3_conv(x)                          #torch.Size([8, 256, 43, 32])
        x = self.proj3_bn(x).reshape(T, B, -1, H//8, W//8).contiguous() #torch.Size([4, 2, 256, 43, 32])

        x = self.proj4_lif(x).flatten(0, 1).contiguous() #torch.Size([8, 256, 43, 32])
        x = self.maxpool4(x)                            #torch.Size([8, 256, 21, 16])
        x = self.proj4_conv(x)                          #torch.Size([8, 256, 21, 16])
        x = self.proj4_bn(x).reshape(T, B, -1, H//16, W//16).contiguous()  #torch.Size([4, 2, 256, 21, 16])
        return x, (None, None)


class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        # print("depths is", depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SpikingTokenizer(img_size_h=img_size_h,
                          img_size_w=img_size_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dims)
        num_patches = patch_embed.num_patches
        block = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        self.Bottleneck_block = nn.ModuleList([BottleneckTransformer(
            dim=embed_dims, num_heads=4, mlp_ratio=mlp_ratios,
            qkv_bias=qkv_bias, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
            for j in range(2)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(8192, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

        self.MST=MST(num_heads=8,clip_n =3,dim=4096, clip_len=16,  init_values=None)

        self.bottleneck_initfeature = nn.Parameter(torch.randn(1, 64, 256))
        kaiming_init(self.bottleneck_initfeature, mode='fan_out')  
        self.pos_embed = nn.Parameter(torch.zeros(1, 400, 256))
        self.pos_drop = nn.Dropout(p=0.)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, (H, W) = patch_embed(x)   #x--torch.Size([4, 2, 256, 21, 16])
        for blk in block:
            x = blk(x)     #torch.Size([16, 8, 256, 21, 16])     #torch.Size([4, 2, 256, 21, 16])  
        T,B,C,H,W =x.shape

        botlk_x = x.flatten(3)   #torch.Size([4, 2, 256, 336])   T, B, C, N
        botlk_x = botlk_x.flatten(0, 1).permute(0,2,1)  #torch.Size([8, 336, 256])

        # bottleneck_feature = self.bottleneck_initfeature.expand_as(botlk_x)
        bottleneck_feature = self.bottleneck_initfeature.expand(T*B,64,C)
        fusion_feature_agg = torch.cat((bottleneck_feature,botlk_x),1)  #BT, 672,256
        fusion_feature_agg = self.pos_drop(fusion_feature_agg + self.pos_embed)  # 

        for bottleneck_blk in self.Bottleneck_block:
            fusion_feature_agg = bottleneck_blk(fusion_feature_agg)
        bottleneck_feature = fusion_feature_agg[:,0:336,:].mean(1).reshape(B,1,-1) #torch.Size([16, 2, 336, 256])
        return x.flatten(3).mean(3), bottleneck_feature   #torch.Size([4, 2, 256])

    def forward(self, event_tensor,rgb_images):
        B, N, C, H, W = event_tensor.shape
        event_tensor = event_tensor.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]    torch.Size([16, 2, 12, 346, 260])
        event_tensor, bottleneck_feature = self.forward_features(event_tensor) #torch.Size([16, 2, 256, 21, 16])   torch.Size([16, 2, 336, 256])
        event_tensor = event_tensor.permute(1,0,2).reshape(B,-1)
        rgb_images = self.MST(rgb_images,bottleneck_feature)           #torch.Size([2, 4096])
        data_out1 = torch.cat((event_tensor,rgb_images),1)
        x1 = self.head(data_out1)

        return x1


@register_model
def Spikingformer(pretrained=False, **kwargs):
    model = vit_snn(
        patch_size=16, embed_dims=256, num_heads=8, mlp_ratios=4,
        in_channels=12, num_classes=300, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=2, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

