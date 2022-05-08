import os
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import math

# ----------------------------------- RCAN ------------------------------------------

def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, opt):
        super(RCAN, self).__init__()
        n_resgroups = opt.n_resgroups
        n_resblocks = opt.n_resblocks
        n_feats = opt.n_feats
        kernel_size = 3
        reduction = opt.reduction
        act = nn.ReLU(True)
        self.narch = opt.narch

        if not opt.norm == None:
            self.normalize, self.unnormalize = normalizationTransforms(opt.norm)
        else:
            self.normalize, self.unnormalize = None, None


        # define head module
        if self.narch == 0:
            modules_head = [conv(opt.nch_in, n_feats, kernel_size)]
            self.head = nn.Sequential(*modules_head)
        else:
            self.head0 = conv(1, n_feats, kernel_size)
            self.head02 = conv(n_feats, n_feats, kernel_size)
            self.head1 = conv(1, n_feats, kernel_size)
            self.head12 = conv(n_feats, n_feats, kernel_size)
            self.head2 = conv(1, n_feats, kernel_size)
            self.head22 = conv(n_feats, n_feats, kernel_size)
            self.head3 = conv(1, n_feats, kernel_size)
            self.head32 = conv(n_feats, n_feats, kernel_size)
            self.head4 = conv(1, n_feats, kernel_size)
            self.head42 = conv(n_feats, n_feats, kernel_size)
            self.head5 = conv(1, n_feats, kernel_size)
            self.head52 = conv(n_feats, n_feats, kernel_size)
            self.head6 = conv(1, n_feats, kernel_size)
            self.head62 = conv(n_feats, n_feats, kernel_size)
            self.head7 = conv(1, n_feats, kernel_size)
            self.head72 = conv(n_feats, n_feats, kernel_size)
            self.head8 = conv(1, n_feats, kernel_size)
            self.head82 = conv(n_feats, n_feats, kernel_size)
            self.combineHead = conv(9*n_feats, n_feats, kernel_size)



        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        if opt.scale == 1:
            modules_tail = [nn.Conv2d(n_feats, opt.nch_out, 1)]
        else:
            modules_tail = [
                Upsampler(conv, opt.scale, n_feats, act=False),
                conv(n_feats, opt.nch_out, kernel_size)]

        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        if not self.normalize == None:
            x = self.normalize(x)

        if self.narch == 0:
            x = self.head(x)
        else:
            x0 = self.head02(self.head0(x[:,0:0+1,:,:]))
            x1 = self.head12(self.head1(x[:,1:1+1,:,:]))
            x2 = self.head22(self.head2(x[:,2:2+1,:,:]))
            x3 = self.head32(self.head3(x[:,3:3+1,:,:]))
            x4 = self.head42(self.head4(x[:,4:4+1,:,:]))
            x5 = self.head52(self.head5(x[:,5:5+1,:,:]))
            x6 = self.head62(self.head6(x[:,6:6+1,:,:]))
            x7 = self.head72(self.head7(x[:,7:7+1,:,:]))
            x8 = self.head82(self.head8(x[:,8:8+1,:,:]))
            x = torch.cat((x0,x1,x2,x3,x4,x5,x6,x7,x8), 1)
            x = self.combineHead(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        if not self.unnormalize == None:
            x = self.unnormalize(x)

        return x


