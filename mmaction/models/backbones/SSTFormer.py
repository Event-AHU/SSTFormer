# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import math
from torch.nn.init import kaiming_normal_, constant_
from .hardutils import predict_flow, crop_like, conv_s, conv, deconv, conv_s_p, conv_ac 

import matplotlib.pyplot as plt 

import torch.nn.functional as F
import numpy as np 
import cv2 
import pdb 

from mmcv.ops import DeformConv2dPack as DCN  
from collections import OrderedDict

import matplotlib.pyplot as plt

from mmaction.registry import MODELS

from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmengine.model.weight_init import constant_init, kaiming_init, normal_init
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

import snntorch as snn

from timm.models.layers import to_2tuple
from mmaction.models.backbones import torchvision_resnet 




@MODELS.register_module()
class SSTFormer(nn.Module):
    expansion = 1
    # default_hyper_params = dict(pretrain_model_path="", crop_pad=4, pruned=True,)
    def __init__(self, pretrained=None, batchNorm=True, output_layers=None, init_std=0.05,num_heads=8,clip_n =3,
                dim=972, clip_len=8, tau: float = 2.,threshold=0.75, init_values=None,to_device="cuda:0",batch_size=5):  
        super(SSTFormer, self).__init__()
        self.batchNorm = batchNorm
        # self.conv1 = conv_s(self.batchNorm,   8,   64, kernel_size=3, stride=1)
        self.conv1 = conv_s(self.batchNorm,   12,   64, kernel_size=3, stride=1)
        self.conv2 = conv_s(self.batchNorm,  64,  64, kernel_size=3, stride=1)
        self.conv3 = conv_s(self.batchNorm, 64,  128, kernel_size=3, stride=1)
        self.conv4 = conv_s(self.batchNorm, 128,  128, kernel_size=3, stride=1)
        self.conv5 = conv_s(self.batchNorm, 128,  256, kernel_size=3, stride=1)
        self.conv6 = conv_s(self.batchNorm, 256,  256, kernel_size=3, stride=1)
        self.conv7 = conv_s(self.batchNorm, 256,  512, kernel_size=3, stride=1)
        self.conv8 = conv_s(self.batchNorm, 512,  512, kernel_size=3, stride=1)

        self.deconv3 = deconv(self.batchNorm, 512,  256)
        self.deconv2 = deconv(self.batchNorm, 512, 128)
        self.deconv1 = deconv(self.batchNorm, 960, 960)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

        self.fusion_feature_map = nn.Parameter(torch.randn(1, 16, 60, 60))
        kaiming_init(self.fusion_feature_map, mode='fan_out')  
        self.fu_concat_1 = conv_s(self.batchNorm, 256,  16, kernel_size=3, stride=1)
        self.fu_concat_2 = nn.Conv2d(in_channels=16,out_channels=256, kernel_size=5, stride=3,padding=1)


        ## DCN-v2: dynamic convolutional network module 
        self.dcnfeat_layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.GroupNorm(32, 32),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('dcn_conv2', nn.Sequential(DCN(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1),
                                        nn.ReLU(inplace=True),
                                        nn.GroupNorm(32, 32),
                                        nn.MaxPool2d(kernel_size=3, stride=2)
                                        )),
                ('dcn_conv3', nn.Sequential(DCN(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1),
                                        nn.ReLU(inplace=True),
                                        nn.GroupNorm(32, 32),
                                        )),
                ('dcn_conv4', nn.Sequential(DCN(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1),
                                        nn.ReLU(inplace=True),
                                        nn.GroupNorm(32, 32))),                                        
                ('dcn_conv5', nn.Sequential(DCN(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1),
                                        nn.ReLU(inplace=True)))
                ]))
        self.former_out = nn.Sequential(
            # nn.Linear(768, 32),
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
        )

        self.snn_former_fc = nn.Sequential(
            # nn.Linear(7936, 4096),
            # nn.Linear(8800, 4096),
            nn.Linear(7232, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.5),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout3d(0.5),
            # nn.Linear(4096, 280)
        )

        self.init_values =init_values
        self.batch_size = batch_size
        self.clip_len =clip_len
        self.GRU_Transformer = GRU_Transformer(clip_n =clip_n, dim=dim,num_heads=num_heads,qkv_bias=False,drop_rate=0.0,init_values=self.init_values)
        
        self.pretrained = pretrained
        self.init_std = init_std 

        self.pool= nn.MaxPool2d(2, 2)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.batchNorm2 = nn.BatchNorm2d(128)
        self.batchNorm3 = nn.BatchNorm2d(256)
        self.batchNorm4 = nn.BatchNorm2d(512)
        beta1 = 0.5


        self.snn_lif1 = snn.Lapicque(beta=beta1,threshold=threshold)
        self.snn_lif2 = snn.Lapicque(beta=beta1, threshold=threshold)
        self.snn_lif3 = snn.Lapicque(beta=beta1, threshold=threshold)
        self.snn_lif4 = snn.Lapicque(beta=beta1, threshold=threshold)
        self.snn_lif5 = snn.Lapicque(beta=beta1, threshold=threshold)
        self.snn_lif6 = snn.Lapicque(beta=beta1, threshold=threshold)
        self.snn_lif7 = snn.Lapicque(beta=beta1, threshold=threshold)
        self.snn_lif8 = snn.Lapicque(beta=beta1, threshold=threshold)


        self.resnet18_feature_extractor = torchvision_resnet.resnet18(pretrained = True)
        # self.resnet50_feature_extractor = torchvision_resnet.resnet50(pretrained = True)
        self.conv2d_query = nn.Conv2d(in_channels=512,out_channels=4096, kernel_size=1, stride=1,padding=0)
        self.conv2d_support = nn.Conv2d(in_channels=512,out_channels=64, kernel_size=1, stride=1,padding=0)


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=self.init_std)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')


    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, input_, output_layers=None, image_resize=240, sp_threshold=0.75):
        rgb_input = input_[0] #torch.Size([6, 16, 260, 346, 3])torch.Size([2, 3, 16, 224, 224])
        input = input_[1] #torch.Size([6, 16, 346, 260, 12])torch.Size([16, 346, 260, 12])

        threshold = sp_threshold

        mem_11 = torch.zeros(input.size(0), 64,  int(math.ceil(image_resize)),  int(math.ceil(image_resize))).cuda()
        mem_12 = torch.zeros(input.size(0), 64,  int(math.ceil(image_resize/2)),  int(math.ceil(image_resize/2))).cuda()

        mem_21 = torch.zeros(input.size(0), 128,  int(math.ceil(image_resize/2)),  int(math.ceil(image_resize/2))).cuda()
        mem_22 = torch.zeros(input.size(0), 128, int(math.ceil(image_resize/4)),  int(math.ceil(image_resize/4))).cuda()

        mem_31 = torch.zeros(input.size(0), 256, int(math.ceil(image_resize/4)),  int(math.ceil(image_resize/4))).cuda()
        mem_32 = torch.zeros(input.size(0), 256, int(math.ceil(image_resize/8)),  int(math.ceil(image_resize/8))).cuda()

        mem_41 = torch.zeros(input.size(0), 512, int(math.ceil(image_resize/8)),  int(math.ceil(image_resize/8))).cuda()
        mem_42 = torch.zeros(input.size(0), 512, int(math.ceil(image_resize/16)),  int(math.ceil(image_resize/16))).cuda()

        mem_1_total = torch.zeros(input.size(0), 64,  int(math.ceil(image_resize/2)),  int(math.ceil(image_resize/2))).cuda()
        mem_2_total = torch.zeros(input.size(0), 128, int(math.ceil(image_resize/4)),  int(math.ceil(image_resize/4))).cuda()
        mem_3_total = torch.zeros(input.size(0), 256, int(math.ceil(image_resize/8)),  int(math.ceil(image_resize/8))).cuda()
        mem_4_total = torch.zeros(input.size(0), 512, int(math.ceil(image_resize/16)),  int(math.ceil(image_resize/16))).cuda()

        input = input.permute(0,1,4,2,3)

        for i in range(input.shape[1]):     ## from 1 to 16 
            
            current_input = input[:, i, :, :, :]  #torch.Size([6, 12, 346, 260])
            current_input = F.interpolate(current_input, [image_resize, image_resize])#torch.Size([6, 12, 240, 240])

            #----------------------------------------------------------------------------

            current_11 = self.conv1(current_input) #torch.Size([6, 64, 240, 240])
            mem_11 = mem_11 + current_11
            mem_11,current_1 =self.snn_lif1(current_11,mem_11) 

            current_1 = self.conv2(current_1)    
            current_1 = self.pool(current_1) #torch.Size([6, 64, 120, 120])
            current_12 = self.batchNorm1(current_1)
            mem_12 = mem_12 + current_12
            mem_1_total = mem_1_total + current_12 
            mem_12,out_conv1 =self.snn_lif2(current_12,mem_12) 

            #----------------------------------------------------------------------------

            current_21 = self.conv3(out_conv1)
            mem_21 = mem_21 + current_21
            mem_21,current_2 =self.snn_lif3(current_21,mem_21)

            current_2 = self.conv4(current_2)    
            current_2 = self.pool(current_2)
            current_22 = self.batchNorm2(current_2)
            mem_22 = mem_22 + current_22
            mem_2_total = mem_2_total + current_22 
            mem_22,out_conv2 =self.snn_lif4(current_22,mem_22)

            #----------------------------------------------------------------------------

            current_31 = self.conv5(out_conv2)
            mem_31 = mem_31 + current_31
            mem_31,current_3 =self.snn_lif5(current_31, mem_31)

            current_3 = self.conv6(current_3)    
            current_3 = self.pool(current_3)
            current_32 = self.batchNorm3(current_3)
            mem_32 = mem_32 + current_32
            mem_3_total = mem_3_total + current_32 
            mem_32,out_conv3 =self.snn_lif6(current_32, mem_32)

            #----------------------------------------------------------------------------

            current_41 = self.conv7(out_conv3)
            mem_41 = mem_41 + current_41
            mem_41, current_4 =self.snn_lif7(current_41,mem_41)

            current_4 = self.conv8(current_4)    
            current_4 = self.pool(current_4)
            current_42 = self.batchNorm4(current_4)
            mem_42 = mem_42 + current_42
            mem_4_total = mem_4_total + current_42 
            mem_42, out_conv4 =self.snn_lif8(current_42, mem_42)
            #----------------------------------------------------------------------------

        out_conv4 = mem_4_total    ## torch.Size([6, 512, 15, 15]) 
        out_conv3 = mem_3_total     #torch.Size([6, 256, 30, 30])
        out_conv2 = mem_2_total     #torch.Size([6, 128, 60, 60])


        out_deconv3 = self.deconv3(out_conv4)  #torch.Size([6, 128, 30, 30])
        concat3 = torch.cat((out_conv3, out_deconv3),1)

        out_deconv2 = self.deconv2(concat3) #torch.Size([6, 128, 60, 60])
        concat2 = torch.cat((out_conv2, out_deconv2),1) #torch.Size([6, 256, 60, 60])

        ############################################################################################################
        concat_fu = self.fu_concat_1(concat2) #torch.Size([6, 16, 60, 60])
        fusion_feature_map = self.fusion_feature_map.expand_as(concat_fu)
        fusion_feature_map_agg = torch.cat((fusion_feature_map,concat_fu),1) #torch.Size([6, 32, 60, 60])

        ############################################################################################################
        fu_eventFeats = self.dcnfeat_layers(fusion_feature_map_agg)   ## torch.Size([6, 32, 14, 14])
        fu_fea_map = fu_eventFeats[:,0:16,:,:] #torch.Size([6, 16, 14, 14])
        scnn_out = fu_eventFeats[:,16:32,:,:] #torch.Size([6, 16, 14, 14])
        scnn_out_Feats = torch.flatten(scnn_out, start_dim=1, end_dim=3) #torch.Size([6, 3136])  ###SCNN_out
        bottleneck_fea = self.fu_concat_2(fu_fea_map).reshape(fu_fea_map.shape[0],-1)
        ################################################################################################
        
        B,C,N,H,W =rgb_input.shape
        rgb_input_res = rgb_input.reshape(B*N,C,H,W)
        rgb_res_out = self.resnet18_feature_extractor(rgb_input_res).reshape( B,N,512,8,8)
        # rgb_res_out = self.resnet50_feature_extractor(rgb_input_res).reshape( B,N,2048,8,8)

        if self.clip_len==16:
            res_query = torch.cat((rgb_res_out[:,3:4,:,:,:],rgb_res_out[:,7:8,:,:,:],rgb_res_out[:,11:12,:,:,:],rgb_res_out[:,15:16,:,:,:]),1).reshape(B*4,512,8,8)#torch.Size([10, 4, 512, 8, 8])
            res_support = torch.cat((rgb_res_out[:,0:3,:,:,:],rgb_res_out[:,4:7,:,:,:],rgb_res_out[:,8:11,:,:,:],rgb_res_out[:,12:15,:,:,:]),1).reshape(B*(N-4),512,8,8)#torch.Size([10, 12, 512, 8, 8])
            former_query = self.conv2d_query(res_query).reshape(B,4,64,4096)  #torch.Size([10, 4, 64, 4096])
            former_support = self.conv2d_support(res_support).reshape(B,N-4,-1)  #torch.Size([10, 12, 4096])

        
        former_out = self.GRU_Transformer(former_support, former_query, bottleneck_fea) #torch.Size([16, 128, 128])
        ################################################################################################
        out_ =  torch.cat((former_out,scnn_out_Feats),1)
        predict = self.snn_former_fc(out_)   ## torch.Size([8, 800, 72, 72])
        # print('self.proj : ',self.snn_former_fc[0].weight)

        return predict 

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def snncnn(output_layers=None, pretrained=False, dilation_factor=1):
    """Constructs a SNN-CNN model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    snn_model = SCNN_TRANSFORMER(batchNorm=False)

    return snn_model


class SpikingNN(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return input.gt(1e-5).type(torch.cuda.FloatTensor)

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 1e-5] = 0
        return grad_input


def IF_Neuron(membrane_potential, threshold):
    
    global threshold_k
    threshold_k = threshold
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold_k, 0)
    membrane_potential = membrane_potential - ex_membrane # hard reset
    # generate spike 
    out = SpikingNN().apply(ex_membrane)
    # out = SpikingNN()(ex_membrane)

    out = out.detach() + (1/threshold)*out - (1/threshold)*out.detach()

    return membrane_potential, out


class InstanceL2Norm(nn.Module):
    """Instance L2 normalization.
    """
    def __init__(self, size_average=True, eps=1e-5, scale=1.0):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
        self.scale = scale

    def forward(self, input):
        if self.size_average:
            return input * (self.scale * ((input.shape[1] * input.shape[2] * input.shape[3]) / (
                        torch.sum((input * input).view(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps)).sqrt())
        else:
            return input * (self.scale / (torch.sum((input * input).view(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps).sqrt())




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


        self.GRU_Attention1=GRU_Attention(dim=dim, num_heads=self.num_heads, mlp_ratio=4.,qkv_bias=False, init_values=1e-5, attn_drop=0., proj_drop=0.,drop_path=0)
        self.GRU_Attention2=GRU_Attention(dim=dim, num_heads=self.num_heads, mlp_ratio=4.,qkv_bias=False, init_values=1e-5, attn_drop=0., proj_drop=0.,drop_path=0)


        self.linear = nn.Linear(dim,64)


    def forward(self, former_support, former_query,bottleneck_fea):
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

            former_query_clip = former_query[:,gap_index:gap_index+1,:,:].squeeze(1)
            former_query_clip = self.pos_drop(former_query_clip + self.pos_embed)
            

            ###########################################################
            if gap_index == (num_gap-1):
                former_support_clip = torch.cat((former_support_clip,bottleneck_fea.unsqueeze(1)),1)
                temple_memcach_attn_out = self.GRU_Attention2(former_support_clip,former_query_clip, temple_memcach)

            else:
                temple_memcach_attn_out = self.GRU_Attention1(former_support_clip,former_query_clip, temple_memcach)

            temple_memcach = self.linear(temple_memcach_attn_out).reshape(temple_memcach_attn_out.shape[0], -1)
        
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
        self.norm4 = nn.LayerNorm(dim)

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
        rgb_former_out = rgb_form + self.drop_path2(self.ls2(self.mlp(self.norm2(rgb_form))))

        return rgb_former_out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        head_dim = 128
        self.scale = head_dim ** -0.5
        self.T = 1  # diffusion epoch
        self.alpha = 0.3  # 0-1

        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mask_ratio = 0.1

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
        # x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
