import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    's': _cfg(crop_pct=0.9),
    'm': _cfg(crop_pct=0.95),
}


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


class Former2Mobile(nn.Module):
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
        self.scale = self.head_dims ** -.5

        self.norm = norm_layer(token_dims)

        self.kv = nn.Linear(token_dims, 2 * num_heads * self.head_dims, bias = qkv_bias)
        self.drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, tokens):
        B, C, H, W = x.shape
        _, N, D = tokens.shape

        res = x

        tokens = self.norm(tokens)
        q = x.reshape(B, self.num_heads, self.head_dims, -1).transpose(-2, -1)
        kv = self.kv(tokens).reshape(B, N, 2, self.num_heads, self.head_dims).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(-2, -1).reshape(B, -1, H, W)
        x = self.drop(x)
        x = self.drop_path(x)
        x = x + res

        return x


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

        """
        self.former2mobile = Former2Mobile(token_dims = token_dims,
                                           in_channels = out_channels,
                                           num_heads = num_heads,
                                           drop = drop,
                                           drop_path = drop_path,
                                           attn_drop = attn_drop,
                                           qkv_bias = qkv_bias,
                                           norm_layer = norm_layer)
        """

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
        #x = self.former2mobile(x, tokens)
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
                 params = None):
        super().__init__()
        self.num_classes = num_classes

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

        self.end_mobile2former = Mobile2Former(token_dims = self.token_dims,
                                               in_channels = params["channel_conv"]["in_channels"],
                                               num_heads = params["block"]["num_heads"][-1],
                                               drop = drop_rate,
                                               drop_path = dpr[-2],
                                               attn_drop = attn_drop,
                                               qkv_bias = qkv_bias,
                                               norm_layer = norm_layer)
        
        self.end_former = Former(token_dims = self.token_dims,
                             mlp_ratio = mlp_ratio,
                             drop = drop_rate,
                             drop_path = dpr[-1],
                             attn_drop = attn_drop,
                             norm_layer = norm_layer,
                             act = act,
                             qkv_bias = qkv_bias)
        
        self.channel_conv = CondConv(params["channel_conv"]["in_channels"], params["channel_conv"]["out_channels"], self.token_dims, 1, 1, 0, K = params["block"]["fc_Ks"][-1])
        self.bn = nn.BatchNorm2d(params["channel_conv"]["out_channels"], eps = 1e-5)
        self.act = act()
        
        self.head = Head(in_channels = params["head"]["in_channels"],
                         token_dims = self.token_dims,
                         hidden_channels = params["head"]["hidden_channels"],
                         num_classes = num_classes,
                         drop = drop_rate)
        self.apply(self._init_weights)

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
    
    def net_update_temperature(self, temperature):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temperature)
    
    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.stem(x)
        x = self.lite_bottleneck(x)

        tokens = self.tokens.repeat(B, 1, 1)
        for block in self.blocks:
            x, tokens = block(x, tokens)
        tokens = self.end_mobile2former(x, tokens)
        tokens = self.end_former(tokens)
        z = tokens[:, 0]
        x = self.channel_conv(x, z)
        x = self.bn(x)
        x = self.act(x)
        x = self.head(x, tokens)

        return x


@register_model
def dmf_96(pretrained = False, **kwargs):
    params= {
        "alpha": 1.0,
        "stem":{
            "out_channels": 12,
            "kernel_size": 3,
        },
        "lite_bneck":{
            "in_channels": 12,
            "hidden_channels": 24,
            "out_channels": 12,
            "fc_groups": 1,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1
        },
        "block":{
            "in_channels": [12, 12, 24, 24, 48, 48, 64, 96],
            "hidden_channels": [36, 72, 72, 144, 192, 288, 384, 576],
            "out_channels": [12, 24, 24, 48, 48, 64, 96, 96],
            "kernel_sizes": [3, 3, 3, 3, 3, 3, 3, 3],
            "strides": [2, 2, 1, 2, 1, 1, 2, 1],
            "paddings": [1, 1, 1, 1, 1, 1, 1, 1],
            "fc_groups": 4,
            "ffn_expansion": 4,
            "fc_Ks": [1, 1, 1, 1, 1, 1, 1, 1],
            "conv_Ks": [4, 4, 4, 4, 4, 4, 8, 8],
            "num_heads": [2, 2, 2, 2, 2, 2, 2, 3],
        },
        "channel_conv":{
            "in_channels": 96,
            "out_channels": 576,
        },
        "head":{
            "in_channels": 576,
            "hidden_channels": 1024
        }
    }

    model = DynamicMobileFormer(drop_rate = 0.2, drop_path_rate = 0.05, attn_drop = 0., params = params)
    model.default_cfg = default_cfgs['s']
    return model


@register_model
def dmf_198(pretrained = False, **kwargs):
    params= {
        "alpha": 1.0,
        "stem":{
            "out_channels": 12,
            "kernel_size": 3,
        },
        "lite_bneck":{
            "in_channels": 12,
            "hidden_channels": 24,
            "out_channels": 12,
            "fc_groups": 1,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1
        },
        "block":{
            "in_channels": [12, 20, 20, 36, 36, 64, 64, 88, 88, 120, 120],
            "hidden_channels": [72, 60, 120, 108, 216, 256, 384, 528, 528, 720, 720], 
            "out_channels": [20, 20, 36, 36, 64, 64, 88, 88, 120, 120, 120],
            "kernel_sizes": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "strides": [2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1],
            "paddings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "fc_groups": 2,
            "ffn_expansion": 3,
            "fc_Ks": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            "conv_Ks": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            "num_heads": [2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4],
        },
        "channel_conv":{
            "in_channels": 120,
            "out_channels": 960,
        },
        "head":{
            "in_channels": 960,
            "hidden_channels": 1600
        }
    }

    model = DynamicMobileFormer(drop_rate = 0.2, drop_path_rate = 0.1, attn_drop = 0., params = params)
    model.default_cfg = default_cfgs['s']
    return model





@register_model
def dmf_288(pretrained = False, **kwargs):
    params= {
        "alpha": 1.0,
        "stem":{
            "out_channels": 16,
            "kernel_size": 3,
        },
        "lite_bneck":{
            "in_channels": 16,
            "hidden_channels": 32,
            "out_channels": 16,
            "fc_groups": 1,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1
        },
        "block":{
            "in_channels": [16, 24, 24, 48, 48, 96, 96, 128, 128, 192, 192],
            "hidden_channels": [96, 96, 144, 192, 288, 384, 576, 768, 768, 1152, 1152],
            "out_channels": [24, 24, 48, 48, 96, 96, 128, 128, 192, 192, 192],
            "kernel_sizes": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "strides": [2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1],
            "paddings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "fc_groups": 4,
            "ffn_expansion": 2.8,
            "fc_Ks": [4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8],
            "conv_Ks": [4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8],
            "num_heads": [2, 2, 2, 2, 3, 3, 4, 4, 4, 6, 6],
        },
        "channel_conv":{
            "in_channels": 192,
            "out_channels": 1152,
        },
        "head":{
            "in_channels": 1152,
            "hidden_channels": 1920
        }
    }

    model = DynamicMobileFormer(drop_rate = 0.3, drop_path_rate = 0.1, attn_drop = 0., params = params)
    model.default_cfg = default_cfgs['s']
    return model



@register_model
def dmf_525(pretrained = False, **kwargs):
    params= {
        "alpha": 1.0,
        "stem":{
            "out_channels": 24,
            "kernel_size": 3,
        },
        "lite_bneck":{
            "in_channels": 24,
            "hidden_channels": 48,
            "out_channels": 24,
            "fc_groups": 1,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1
        },
        "block":{
            "in_channels": [24, 40, 40, 72, 72, 128, 128, 176, 176, 240, 240],
            "hidden_channels": [144, 120, 240, 216, 432, 512, 768, 1056, 1056, 1440, 1440],
            "out_channels": [40, 40, 72, 72, 128, 128, 176, 176, 240, 240, 240],
            "kernel_sizes": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "strides": [2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1],
            "paddings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "fc_groups": 4,
            "ffn_expansion": 3,
            "fc_Ks": [4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8],
            "conv_Ks": [4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8],
            "num_heads": [2, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8],
        },
        "channel_conv":{
            "in_channels": 240,
            "out_channels": 1440,
        },
        "head":{
            "in_channels": 1440,
            "hidden_channels": 1920
        }
    }

    model = DynamicMobileFormer(tokens = [6, 192], drop_rate = 0.3, drop_path_rate = 0.1, attn_drop = 0., params = params)
    model.default_cfg = default_cfgs['s']
    return model

@register_model
def dmf_500(pretrained = False, **kwargs):
    params= {
        "alpha": 1.0,
        "stem":{
            "out_channels": 24,
            "kernel_size": 3,
        },
        "lite_bneck":{
            "in_channels": 24,
            "hidden_channels": 48,
            "out_channels": 24,
            "fc_groups": 1,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1
        },
        "block":{
            "in_channels": [24, 40, 40, 72, 72, 128, 128, 176, 176, 240, 240],
            "hidden_channels": [144, 120, 240, 216, 432, 512, 768, 1056, 1056, 1440, 1440],
            "out_channels": [40, 40, 72, 72, 128, 128, 176, 176, 240, 240, 240],
            "kernel_sizes": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "strides": [2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1],
            "paddings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "fc_groups": 4,
            "ffn_expansion": 2.7,
            "fc_Ks": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            "conv_Ks": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            "num_heads": [2, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8],
        },
        "channel_conv":{
            "in_channels": 240,
            "out_channels": 1440,
        },
        "head":{
            "in_channels": 1440,
            "hidden_channels": 1920
        }
    }

    model = DynamicMobileFormer(tokens = [6, 192], drop_rate = 0.3, drop_path_rate = 0.1, attn_drop = 0., params = params)
    model.default_cfg = default_cfgs['s']
    return model

@register_model
def dmf_615(pretrained = False, **kwargs):
    params= {
        "alpha": 1.0,
        "stem":{
            "out_channels": 24,
            "kernel_size": 3,
        },
        "lite_bneck":{
            "in_channels": 24,
            "hidden_channels": 48,
            "out_channels": 24,
            "fc_groups": 1,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1
        },
        "block":{
            "in_channels": [24, 40, 40, 72, 72, 128, 128, 176, 176, 240, 240, 256],
            "hidden_channels": [144, 120, 240, 216, 432, 512, 768, 1056, 1056, 1440, 1440, 1536],
            "out_channels": [40, 40, 72, 72, 128, 128, 176, 176, 240, 240, 256, 256],
            "kernel_sizes": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "strides": [2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1],
            "paddings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "fc_groups": 4,
            "ffn_expansion": 3.5,
            "fc_Ks": [4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8],
            "conv_Ks": [4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8],
            "num_heads": [2, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8],
        },
        "channel_conv":{
            "in_channels": 256,
            "out_channels": 1440,
        },
        "head":{
            "in_channels": 1440,
            "hidden_channels": 1920
        }
    }

    model = DynamicMobileFormer(tokens = [6, 192], drop_rate = 0.3, drop_path_rate = 0.15, attn_drop = 0., params = params)
    model.default_cfg = default_cfgs['s']
    return model