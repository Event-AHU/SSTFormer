# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import math
from torch.nn.init import kaiming_normal_, constant_
from hardutils import predict_flow, crop_like, conv_s, conv, deconv, conv_s_p, conv_ac 

import matplotlib.pyplot as plt 

import torch.nn.functional as F
import numpy as np 
import cv2 
import pdb 

from mmcv.ops import DeformConv2dPack as DCN  
from collections import OrderedDict

import matplotlib.pyplot as plt

from mmcv.utils import _BatchNorm
from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

from timm.models.layers import to_2tuple
import torchvision_resnet 


class MST(nn.Module):
    expansion = 1
    # default_hyper_params = dict(pretrain_model_path="", crop_pad=4, pruned=True,)
    def __init__(self,  num_heads=8,clip_n =3,dim=972, clip_len=8,  init_values=None):  
        super(MST, self).__init__()

        # self.resnet18_feature_extractor = torchvision_resnet.resnet18(pretrained = True)
        self.resnet50_feature_extractor = torchvision_resnet.resnet50(pretrained = True)
        # self.conv2d_query = nn.Conv2d(in_channels=512,out_channels=4096, kernel_size=1, stride=1,padding=0)
        # self.conv2d_support = nn.Conv2d(in_channels=512,out_channels=64, kernel_size=1, stride=1,padding=0)
        self.conv2d_query = nn.Conv2d(in_channels=2048,out_channels=4096, kernel_size=1, stride=1,padding=0)
        self.conv2d_support = nn.Conv2d(in_channels=2048,out_channels=64, kernel_size=1, stride=1,padding=0)

        self.init_values =init_values
      
        self.clip_len =clip_len
        self.GRU_Transformer = GRU_Transformer(clip_n =clip_n, dim=dim,num_heads=num_heads,qkv_bias=False,drop_rate=0.0,init_values=self.init_values)
 

    def forward(self, rgb_input,bottleneck_feature, image_resize=240, sp_threshold=0.75):

        B,N,C,H,W =rgb_input.shape
        rgb_input_res = rgb_input.reshape(B*N,C,H,W)
        # rgb_res_out = self.resnet18_feature_extractor(rgb_input_res).reshape( B,N,512,8,8)#torch.Size([24, 512, 8, 8])
        rgb_res_out = self.resnet50_feature_extractor(rgb_input_res).reshape( B,N,2048,8,8)#torch.Size([24, 512, 8, 8])
        if self.clip_len==16:
            res_query = torch.cat((rgb_res_out[:,3:4,:,:,:],rgb_res_out[:,7:8,:,:,:],rgb_res_out[:,11:12,:,:,:],rgb_res_out[:,15:16,:,:,:]),1).reshape(B*4,2048,8,8)#torch.Size([10, 4, 512, 8, 8])
            res_support = torch.cat((rgb_res_out[:,0:3,:,:,:],rgb_res_out[:,4:7,:,:,:],rgb_res_out[:,8:11,:,:,:],rgb_res_out[:,12:15,:,:,:]),1).reshape(B*(N-4),2048,8,8)#torch.Size([10, 12, 512, 8, 8])
            former_query = self.conv2d_query(res_query).reshape(B,4,64,4096)  #torch.Size([10, 4, 64, 4096])
            former_support = self.conv2d_support(res_support).reshape(B,N-4,-1)  #torch.Size([10, 12, 4096])
        
        former_out = self.GRU_Transformer(former_support,bottleneck_feature, former_query) #torch.Size([16, 128, 128])
        
        return former_out




class GRU_Transformer(nn.Module):
    def __init__(self,clip_n = 3, dim=4096, num_heads=8,qkv_bias=False,drop_rate=0.0, init_values=1e-5):
        super().__init__()
   
        self.clip_n = clip_n
        self.num_heads = num_heads

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, 4096))

        depth=2
        drop = 0
        drop_path=0.
        attn_drop=0.
        mlp_ratio=4.
        act_layer=nn.GELU
        self.init_values=init_values


        self.GRU_Attention=GRU_Attention(dim=dim, num_heads=self.num_heads, mlp_ratio=4.,qkv_bias=False, init_values=1e-5, attn_drop=0., proj_drop=0.,drop_path=0)



    def forward(self, former_support,bottleneck_feature, former_query):
        #rgb_input.shape #torch.Size([10, 8, 4096])
        support_B,support_N, support_dim = former_support.shape #torch.Size([10, 12, 4096])
        # query_B,query_N_Clip, query_N_Token, query_dim = former_query.shape   #torch.Size([10, 4, 64, 4096])
        clip_gap = support_N /self.clip_n

        num_gap = int(clip_gap)

        for gap_index in range(num_gap):
            if gap_index==0:
                temple_memcach = former_support[:, 0, :]
            start_idx=gap_index*self.clip_n
            end_idx=(gap_index+1)*self.clip_n
            former_support_clip = former_support[:, start_idx:end_idx, :]#torch.Size([B, self.clip_n, 4096])
            former_support_botlk = torch.cat((former_support_clip,bottleneck_feature),1)

            former_query_clip = former_query[:,gap_index:gap_index+1,:,:].squeeze(1)
            former_query_clip = self.pos_drop(former_query_clip + self.pos_embed)   #torch.Size([2, 64, 4096])

            ###########################################################
            temple_memcach_attn_out = self.GRU_Attention(former_support_botlk,former_query_clip, temple_memcach) #torch.Size([2, 64, 4096])
            temple_memcach = temple_memcach_attn_out.mean(1)
        
        return temple_memcach


class GRU_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.,qkv_bias=False, init_values=1e-5,attn_drop=0., proj_drop=0.,drop_path=0):
        super().__init__()

        self.num_heads=num_heads
        self.dim=dim
        act_layer=nn.GELU
        drop_rate=0.0
        # compress_rate = 4
        # self.compress_dim = dim//compress_rate
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=0)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features= self.dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_rate)

        self.GRU = nn.GRU(input_size=self.dim ,hidden_size=self.dim ,num_layers=1,bidirectional=False,batch_first=True)  #(batch,seq,feature)



    def forward(self, former_support_clip,former_query_clip, temple_memcach):
        temple_kv = torch.cat((temple_memcach.unsqueeze(1),former_support_clip),1) #torch.Size([10, 4, 4096])

        temple_q = former_query_clip #torch.Size([10, 64, 4096])


        temple_kv, _  = self.GRU(temple_kv)  #torch.Size([10, 4, 4096])


        rgb_form = temple_q + self.drop_path1(self.ls1(self.attn(self.norm1(temple_q), self.norm2(temple_kv))))  
        rgb_former_out = rgb_form + self.drop_path2(self.ls2(self.mlp(self.norm3(rgb_form))))

        return rgb_former_out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # head_dim = 128
        self.scale = head_dim ** -0.5
        self.T = 1  # diffusion epoch
        self.alpha = 0.3  # 0-1

        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv):
        B_kv, N_kv, C_kv = kv.shape
        kv = self.kv(kv).reshape(B_kv, N_kv, 2, self.num_heads, C_kv // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # torch.Size([10, 8, 4, 512])
                              #q.shape torch.Size([10, 64, 4096])

        q = q.reshape(q.shape[0],self.num_heads, q.shape[1], q.shape[2] // self.num_heads) #torch.Size([10, 8, 64, 512])
        B, N = q.shape[0],q.shape[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  #torch.Size([10, 8, 64, 4])
        attn = attn.softmax(dim=-1)#torch.Size([4, 12, 785, 785])
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma