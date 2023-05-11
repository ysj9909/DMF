import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple


from mmdet.models.builder import BACKBONES 
from mmdet.utils import get_root_logger
from mmcv.runner import _load_checkpoint


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class MLP(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio = 2,
                 drop = 0.,
                 act = nn.GELU):
        super().__init__()
        exp_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, exp_dim, bias = False)
        self.fc2 = nn.Linear(exp_dim, dim, bias = False)

        self.act = act()
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Token_Attention(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads = 8, 
                 qkv_bias = False, 
                 qk_scale = None,
                 attn_drop = 0., 
                 proj_drop = 0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Stem(nn.Module):
    """Stem
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, 
                 in_channels,
                 hidden_channels,
                 out_channels,
                 fc_groups = 1,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 drop_path = 0.,
                 act = nn.GELU,
                 is_grn = False,
                 hidden_shortcut = False):
        super().__init__()
        self.is_grn = is_grn
        hidden_channels = int(hidden_channels)
        self.hidden_shortcut = hidden_shortcut

        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias = False, groups = fc_groups)
        self.bn1 = nn.BatchNorm2d(hidden_channels, eps = 1e-5)

        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, padding, bias = False, groups = hidden_channels)
        if is_grn:
            self.grn = GRN(hidden_channels)
        else:
            self.bn2 = nn.BatchNorm2d(hidden_channels, eps = 1e-5)

        self.pw_conv2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias = False, groups = fc_groups)
        self.bn3 = nn.BatchNorm2d(out_channels, eps = 1e-5)

        self.act = act()

    def forward(self, x):
        x = self.pw_conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        if self.hidden_shortcut:
            x = self.dw_conv(x) + x
        else:
            x = self.dw_conv(x)
        if self.is_grn:
            x = self.grn(x)
        else:
            x = self.bn2(x)
        x = self.act(x)

        x = self.pw_conv2(x)
        x = self.bn3(x)

        return x


class Head(nn.Module):
    def __init__(self, 
                 in_channels,
                 token_dims,
                 hidden_channels,
                 num_classes = 1000,
                 drop = 0.,
                 act = nn.Hardswish):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_features = in_channels + token_dims,
                             out_features = hidden_channels)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(in_features = hidden_channels,
                             out_features = num_classes)
        
        self.act = act()
    
    def forward(self, x, tokens):
        x = self.avgpool(x).reshape(x.shape[0], -1)
        z = tokens[:, 0]
        x = torch.cat([x, z], dim = -1)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


class KernelAttention(nn.Module):
    def __init__(self,in_channels, token_dims, K):
        super().__init__()
        self.temperature = 1.0
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net1 = nn.Conv2d(in_channels + token_dims, token_dims, kernel_size = 1, bias = False)
        self.bn = nn.BatchNorm2d(token_dims, eps = 1e-5)
        self.relu = nn.ReLU()
        self.net2 = nn.Conv2d(token_dims, K, kernel_size = 1, bias = False)
    
    def update_temperature(self, temperature):
        self.temperature = temperature
        

    def forward(self, x, z):
        """
        x : mobile feature map, shape of [B, C, H, W]
        z : average pooled Former feature vector, shape of [B, C]
        """
        att = z.unsqueeze(-1).unsqueeze(-1)
        avg_pool = self.avg_pool(x)
        att = torch.cat([att, avg_pool], dim = 1)
        att = self.relu(self.bn(self.net1(att)))
        att=self.net2(att).reshape(x.shape[0],-1) #bs,K
        #return F.softmax(att / self.temperature, dim = -1)
        return torch.sigmoid(att / self.temperature)


class CondConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 token_dims,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 dilation = 1,
                 groups = 1,
                 bias = False,
                 K = 4,
                 init_weight = True): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.Kernel_Attention = KernelAttention(in_channels, token_dims, K)

        self.static_weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size), requires_grad = True)
        self.weight = nn.Parameter(torch.zeros(K,out_channels, in_channels // groups, kernel_size, kernel_size), requires_grad = True)
        if(bias):
            self.static_bias = nn.Parameter(torch.randn(out_channels), requires_grad = True)
            self.bias = nn.Parameter(torch.randn(K,out_channels), requires_grad=True)
        else:
            self.bias = None

        if(init_weight):
            self._initialize_weights()

        
    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.static_weight)
    
    def update_temperature(self, temperature):
        self.Kernel_Attention.update_temperature(temperature)
    

    def forward(self, x, z):
        bs,_,h,w=x.shape
        softmax_att=self.Kernel_Attention(x, z) #bs,K
        x=x.reshape(1, -1, h, w) #1,bs*in_f,h,w
        weight=self.weight.reshape(self.K,-1) * 0.1 #K,-1
        aggregate_weight=torch.mm(softmax_att,weight).reshape(bs , self.out_channels,self.in_channels // self.groups,
                                self.kernel_size,self.kernel_size) #bs,out_f,in_f,k,k
        aggregate_weight = aggregate_weight  + self.static_weight.unsqueeze(0)
        aggregate_weight = aggregate_weight.reshape(bs*self.out_channels, self.in_channels//self.groups, 
                                self.kernel_size, self.kernel_size)

        if(self.bias is not None):
            bias=self.bias.reshape(self.K,-1) #K,out_p
            aggregate_bias=torch.mm(softmax_att,bias).reshape(bs, self.out_channels) #bs,out_p
            aggregate_bias = aggregate_bias + self.static_bias.unsqueeze(0)
            aggregate_bias = aggregate_bias.reshape(-1)
            output=F.conv2d(x,weight=aggregate_weight,bias=aggregate_bias,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)
        else:
            output=F.conv2d(x,weight=aggregate_weight,bias=None,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)
        
        output=output.reshape(bs,self.out_channels,h,w)
        return output




class DynamicMobile(nn.Module):
    def __init__(self, 
                 in_channels,
                 hidden_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 fc_groups = 1,
                 token_dims = None,
                 fc_K = 4,
                 conv_K = 16):
        super().__init__()
        self.is_downsample = True if stride == 2 else False
        self.fc_K = fc_K
        self.conv_K = conv_K
        if self.is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups = in_channels),
                nn.BatchNorm2d(in_channels, eps = 1e-5),
                nn.GELU(),
            )
        if fc_K > 1:
            self.pw_conv1 = CondConv(in_channels, hidden_channels, token_dims, 1, 1, 0, groups = fc_groups, K = fc_K)
        else:
            self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, groups = fc_groups, bias = False)
        self.bn1 = nn.BatchNorm2d(hidden_channels, eps = 1e-5)
        
        if conv_K > 1:
            self.dw_conv = CondConv(hidden_channels, hidden_channels, token_dims, kernel_size, 1, padding, groups = hidden_channels, K = conv_K)
        else:
            self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, padding, groups = hidden_channels, bias  =False)
        self.bn2 = nn.BatchNorm2d(hidden_channels, eps = 1e-5)

        if fc_K > 1:
            self.pw_conv2 = CondConv(hidden_channels, out_channels, token_dims, 1, 1, 0, groups = fc_groups, K = fc_K)
        else:
            self.pw_conv2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, groups = fc_groups , bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps = 1e-5)
        self.act = nn.GELU()

    def forward(self, x, z):
        if self.is_downsample:
            x = self.downsample(x)
        if self.fc_K > 1:
            x = self.pw_conv1(x, z)
        else:
            x = self.pw_conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        if self.conv_K > 1:
            x = self.dw_conv(x, z)
        else:
            x = self.dw_conv(x)
        x = self.bn2(x)
        x = self.act(x)
        if self.fc_K > 1:
            x = self.pw_conv2(x, z)
        else:
            x = self.pw_conv2(x)
        x = self.bn3(x)
        
        return x


class Former(nn.Module):
    def __init__(self, 
                 token_dims,
                 mlp_ratio = 2,
                 drop = 0.,
                 drop_path = 0.,
                 attn_drop = 0.,
                 norm_layer = nn.LayerNorm,
                 act = nn.GELU,
                 qkv_bias = False):
        super().__init__()
        num_heads = int(token_dims // 32)

        self.token_attention = Token_Attention(token_dims, num_heads, qkv_bias, attn_drop = attn_drop, proj_drop = drop)
        self.norm1 = norm_layer(token_dims)
        self.drop_path1 = DropPath(drop_path)

        self.mlp = MLP(token_dims, mlp_ratio, drop, act)
        self.norm2 = norm_layer(token_dims)
        self.drop_path2 = DropPath(drop_path)
    
    def forward(self, x):
        x = x + self.drop_path1(self.token_attention(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class Mobile2Former(nn.Module):
    # fourier positional encoding 넣을 수 있는 방법 생각해보자
    def __init__(self,
                 token_dims, 
                 in_channels,
                 num_heads = 1,
                 drop = 0.,
                 drop_path = 0.,
                 attn_drop = 0.,
                 qkv_bias = False,
                 norm_layer = nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.head_dims = in_channels // num_heads
        self.scale = self.head_dims ** -0.5
        #self.fourier_encode = fourier_encode

        self.norm = norm_layer(token_dims)
        self.q = nn.Linear(token_dims, num_heads * self.head_dims, bias = qkv_bias)
        self.fc = nn.Linear(in_channels, token_dims, bias = False)
        self.drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, tokens):
        B, C, H, W = x.shape
        _, N, D = tokens.shape
        tokens = self.norm(tokens)

        x = x.reshape(B, self.num_heads, self.head_dims, -1).transpose(-2, -1)
        q = self.q(tokens).reshape(B, N, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        
        attn = (q @ x.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        z = (attn @ x).transpose(1, 2).reshape(B, N, -1)
        z = self.fc(z)
        z = self.drop(z)
        z = self.drop_path(z)
        z = z + tokens
        return z



class DMFBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 hidden_channels,
                 out_channels,
                 token_dims,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 fc_groups = 1,
                 ffn_expansion = 3,
                 fc_K = 4, 
                 conv_K = 16,
                 num_heads = 1,
                 mlp_ratio = 2,
                 drop = 0.,
                 attn_drop = 0.,
                 drop_path = 0.,
                 norm_layer = nn.LayerNorm,
                 act = nn.GELU,
                 qkv_bias = False):
        super().__init__()
        self.drop_path = DropPath(drop_path)
        self.norm = GroupNorm(out_channels)

        self.dynamic_mobile = DynamicMobile(in_channels = in_channels,
                                            hidden_channels = hidden_channels,
                                            out_channels = out_channels,
                                            kernel_size = kernel_size,
                                            stride = stride,
                                            padding = padding,
                                            fc_groups = fc_groups,
                                            token_dims = token_dims,
                                            fc_K = fc_K,
                                            conv_K = conv_K)
        

        self.former = Former(token_dims = token_dims,
                             mlp_ratio = mlp_ratio,
                             drop = drop,
                             drop_path = drop_path,
                             attn_drop = attn_drop,
                             norm_layer = norm_layer,
                             act = act,
                             qkv_bias = qkv_bias)
        

        self.mobile2former = Mobile2Former(token_dims = token_dims,
                                           in_channels = in_channels,
                                           num_heads = num_heads,
                                           drop = drop,
                                           drop_path = drop_path,
                                           attn_drop = attn_drop,
                                           qkv_bias = qkv_bias,
                                           norm_layer = norm_layer)

        self.ffn = BottleNeck(in_channels = out_channels,
                              hidden_channels = out_channels * ffn_expansion,
                              out_channels = out_channels,
                              drop_path = drop_path,
                              act = act,
                              is_grn = True,
                              hidden_shortcut = True)

    
    def forward(self, x, tokens):
        tokens = self.mobile2former(x, tokens)
        tokens = self.former(tokens)

        z = tokens[:, 0]
        x = self.dynamic_mobile(x, z)
        x = x + self.drop_path(self.ffn(self.norm(x)))

        return x, tokens


class DynamicMobileFormer(nn.Module):
    def __init__(self,
                 num_classes = 1000,
                 in_chans = 3,
                 tokens = [6, 192],
                 mlp_ratio = 2,
                 drop_rate = 0.,
                 drop_path_rate = 0., 
                 attn_drop = 0.,
                 qkv_bias = False,
                 norm_layer = nn.LayerNorm,
                 act = nn.GELU,
                 out_indices = (0, 1, 2, 3),
                 frozen_stages = 1,
                 norm_eval = True,
                 pretrained = None,
                 init_cfg = None,
                 norm_cfg = None,
                 style = None,
                 depth = None,
                 num_stages = None):
        super().__init__()
        params= {
            "stage_idx": [1, 3, 7, 10],
            "alpha": 1.0,
            "stem":{
                "out_channels": 16,
                "kernel_size": 3,
            },
            "lite_bneck":{
                "in_channels": 16,
                "hidden_channels": 32,
                "out_channels": 18,
                "fc_groups": 1,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1
            },
            "block":{
                "in_channels": [18, 30, 30, 54, 54, 96, 96, 132, 132, 180, 180],
                "hidden_channels": [108, 90, 180, 162, 324, 384, 576, 792, 792, 1080, 1080],
                "out_channels": [30, 30, 54, 54, 96, 96, 132, 132, 180, 180, 180], 
                "kernel_sizes": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                "strides": [2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1],
                "paddings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "fc_groups": 3,
                "ffn_expansion": 2,
                "fc_Ks": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                "conv_Ks": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                "num_heads": [2, 2, 2, 2, 2, 3, 3, 4, 4, 6, 6],
            },
            "channel_conv":{
                "in_channels": 180,
                "out_channels": 1152,
            },
            "head":{
                "in_channels": 1152,
                "hidden_channels": 1920
            }
        }
        self.stage_idx = params["stage_idx"]
        self.out_indices = [self.stage_idx[x] for x in out_indices]
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages

        self.num_tokens, self.token_dims = tokens[0], tokens[1]
        # create learnable tokens
        self.tokens = nn.Parameter(nn.Parameter(torch.randn(1, self.num_tokens, self.token_dims)))
        alpha = params["alpha"]

        self.stem = Stem(in_channels = in_chans,
                         out_channels = int(params["stem"]["out_channels"] * alpha),
                         kernel_size = params["stem"]["kernel_size"],
                         stride = 2,
                         padding = params["stem"]["kernel_size"] // 2,
                         act = act)
        
        self.lite_bottleneck = BottleNeck(in_channels = int(params["lite_bneck"]["in_channels"] * alpha),
                                          hidden_channels = int(params["lite_bneck"]["hidden_channels"] * alpha),
                                          out_channels = int(params["lite_bneck"]["out_channels"] * alpha),
                                          fc_groups = params["lite_bneck"]["fc_groups"],
                                          kernel_size = params["lite_bneck"]["kernel_size"],
                                          stride = params["lite_bneck"]["stride"],
                                          padding = params["lite_bneck"]["padding"])
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(params["block"]["in_channels"]) + 2)]  # stochastic depth decay rule
        self.blocks = []
        for i, _ in enumerate(params["block"]["in_channels"]):
            self.blocks.append(
                DMFBlock(
                    in_channels = int(params["block"]["in_channels"][i] * alpha),
                    hidden_channels = int(params["block"]["hidden_channels"][i] * alpha),
                    out_channels = int(params["block"]["out_channels"][i] * alpha),
                    token_dims = self.token_dims,
                    kernel_size = params["block"]["kernel_sizes"][i],
                    stride = params["block"]["strides"][i],
                    padding = params["block"]["paddings"][i],
                    fc_groups = params["block"]["fc_groups"],
                    ffn_expansion = params["block"]["ffn_expansion"],
                    fc_K = params["block"]["fc_Ks"][i],
                    conv_K = params["block"]["conv_Ks"][i],
                    num_heads = params["block"]["num_heads"][i],
                    mlp_ratio = mlp_ratio,
                    drop = drop_rate,
                    attn_drop = attn_drop,
                    drop_path = dpr[i],
                    norm_layer = norm_layer,
                    act = act,
                    qkv_bias = qkv_bias
                )
            )
        self.blocks = nn.ModuleList(self.blocks)

        self.apply(self._init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        self.init_weights()
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.train()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        self.net_update_temperature(1.0)
    
    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
    
    def net_update_temperature(self, temperature):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temperature)


    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.stage_idx[self.frozen_stages]+1):
                m = self.blocks[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(DynamicMobileFormer, self).train(mode)
        self._freeze_stages()

        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


    def forward(self, x):
        outs = []
        B, _, _, _ = x.shape
        x = self.stem(x)
        x = self.lite_bottleneck(x)

        tokens = self.tokens.repeat(B, 1, 1)
        for idx, block in enumerate(self.blocks):
            x, tokens = block(x, tokens)
            if idx in self.out_indices:
                outs.append(x)

        return outs


@BACKBONES.register_module()
class dmf_499(DynamicMobileFormer):
    def __init__(self, **kwargs):
        super(dmf_499, self).__init__(
            tokens = [6, 192], mlp_ratio = 2, drop_rate = 0., drop_path_rate = 0.1, attn_drop = 0.,
            qkv_bias = False, norm_layer = nn.LayerNorm, act = nn.GELU, **kwargs)


@BACKBONES.register_module()
class dmf_285(DynamicMobileFormer):
    def __init__(self, **kwargs):
        super(dmf_285, self).__init__(
            tokens = [6, 192], mlp_ratio = 2, drop_rate = 0., drop_path_rate = 0.1, attn_drop = 0.,
            qkv_bias = False, norm_layer = nn.LayerNorm, act = nn.GELU, **kwargs)


@BACKBONES.register_module()
class dmf_198(DynamicMobileFormer):
    def __init__(self, **kwargs):
        super(dmf_198, self).__init__(
            tokens = [6, 192], mlp_ratio = 2, drop_rate = 0., drop_path_rate = 0.1, attn_drop = 0.,
            qkv_bias = False, norm_layer = nn.LayerNorm, act = nn.GELU, **kwargs)

