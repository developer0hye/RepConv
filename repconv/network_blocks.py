import torch
import torch.nn as nn
import copy

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    conv = copy.deepcopy(conv)
    bn = copy.deepcopy(bn)
    
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)
    
    conv_weight = conv.weight #shape: [out_c, in_c, h, w]
    conv_bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias #shape: [out_c]
    bn_affine_weight = torch.ones(conv.weight.shape[0], device=conv.weight.device) if bn.weight is None else bn.weight #shape: [out_c]
    bn_affine_bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if bn.bias is None else bn.bias #shape: [out_c]
    bn_running_mean = bn.running_mean #shape: [out_c]
    bn_running_std = torch.sqrt(bn.eps + bn.running_var) #shape: [out_c]
    
    fusedconv.weight.copy_((conv_weight * bn_affine_weight.view(-1, 1, 1, 1)) / bn_running_std.view(-1, 1, 1, 1))
    fusedconv.bias.copy_(((conv_bias-bn_running_mean)/bn_running_std) + bn_affine_bias)
    return fusedconv

class ConvBatchNorm2d(nn.Module):
    def __init__(
        self, 
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1
        )
        self.bn = nn.BatchNorm2d(64)
        
        self.bn.running_mean.copy_(torch.randn_like(self.bn.running_mean))
        self.bn.running_var.copy_(abs(torch.randn_like(self.bn.running_var))+1e-6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
    def fuse_forward(self, x):
        fused_conv = fuse_conv_and_bn(self.conv, self.bn)
        fused_conv.eval()
        
        return fused_conv(x)

model = ConvBatchNorm2d()
input = torch.randn((1, 3, 224, 224))
model.eval()

with torch.no_grad():
    print(model(input).sum())
    print(model.fuse_forward(input).sum())


# class RepConv2d(nn.Module):
#     def __init__(
#         self, 
#         in_channels,
#         out_channels,
#         kernel_size=3,
#         stride=1,
#         padding=1,
#         groups=1,
#         bias=False,
#     ):
#         super().__init__()
#         # same padding
        
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             groups=groups,
#             bias=bias,
#         )
#         self.conv_bn = nn.BatchNorm2d(out_channels)
        
#         self.pointwise_conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=1,
#             stride=stride,
#             padding=padding-kernel_size//2,
#             groups=groups,
#             bias=bias,
#         )
#         self.pointwise_conv_bn = nn.BatchNorm2d(out_channels)

#         self.identity = nn.BatchNorm2d(in_channels) if stride == 1 and in_channels == out_channels else False
#         self.re_parameterized = False
    
#     def identity2conv(self, identity):
#         assert isinstance(identity, nn.BatchNorm2d)
#         identity_conv = nn.Conv2d(
#             self.conv.in_channels,
#             self.conv.out_channels,
#             kernel_size=self.conv.kernel_size,
#             stride=self.conv.stride,
#             padding=self.conv.padding,
#             groups=self.conv.groups,
#             bias=True,
#         )
        
#         kernel_h, kernel_w = identity_conv.kernel_size
#         weight = torch.zeros_like(identity_conv.weight)
#         in_channels_per_group = self.conv.in_channels // self.conv.groups
#         for i in range(self.conv.in_channels):
#             weight[i, i % in_channels_per_group, 1, 1] = 1
        
#         print(weight[:, :, kernel_h//2, kernel_w//2].shape)
        
#         # weight = torch.zeros_like(identity_conv.weight)
#         # weight[:, :, kernel_h//2, kernel_w//2] = conv.weight.flatten(1)
        
#         identity_conv = fuse_conv_and_bn(identity_conv, identity)
#         return identity_conv
    
#     def pad_pointwise_conv(self, conv):
#         assert conv.kernel_size == (1, 1)
#         assert self.conv.kernel_size[0] % 2 == 1 and self.conv.kernel_size[1] % 2 == 1
        
#         padded_pointwise_conv = nn.Conv2d(
#             self.conv.in_channels,
#             self.conv.out_channels,
#             kernel_size=self.conv.kernel_size,
#             stride=self.conv.stride,
#             padding=self.conv.padding,  
#             groups=self.conv.groups,
#             bias=False if conv.bias is None else True,
#         ).to(self.conv.weight.device)
        
#         kernel_h, kernel_w = padded_pointwise_conv.kernel_size
        
#         padded_weight = torch.zeros_like(padded_pointwise_conv.weight)
#         padded_weight[:, :, kernel_h//2, kernel_w//2] = conv.weight.flatten(1)
        
#         padded_pointwise_conv.weight.copy_(padded_weight)
#         padded_pointwise_conv.bias.copy_(conv.bias)
#         return padded_pointwise_conv
    
#     @torch.no_grad()
#     def re_parameterization(self):
#         bn_fused_conv = fuse_conv_and_bn(self.conv, self.conv_bn)
#         bn_fused_pointwise_conv = fuse_conv_and_bn(self.pointwise_conv, self.pointwise_conv_bn)
#         bn_fused_pointwise_conv = self.pad_pointwise_conv(bn_fused_pointwise_conv)
        
#         if self.identity:
#             identity_conv = self.identity2conv(self.identity)
            
#             pass
        
        
#         self.re_parameterized = True
        
        

#     def forward(self, x):
#         if self.training:
#             y_0 = self.conv_bn(self.conv(x))
#             y_1 = self.pointwise_conv_bn(self.pointwise_conv(x))
#             y = y_0 + y_1
#             if self.identity: 
#                 y += self.identity(x)
#             return y

#         if not self.re_parameterized:
#             self.re_parameterization()
#         y = self.conv(x)
#         return y

# model = RepConv2d(in_channels=64, out_channels=64, groups=2)
# model.re_parameterization()