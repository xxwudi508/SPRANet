#!/usr/bin/python3
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pvtv2 import pvt_v2_b2
from swin import SwinTransformer
from resnet_encoder import ResNet
import sys


sys.path.append('./')

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class Res2NetBlock(nn.Module):
    def __init__(self, outplanes, inchannel, scales=3):
        super(Res2NetBlock, self).__init__()

        self.scales = scales
        # 1*1的卷积层
        self.inconv = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.BatchNorm2d(outplanes),
        )

        # 3*3的卷积层，一共有3个卷积层和3个BN层
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(outplanes // 2, 64, 1),
        #     nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False),
        #     nn.BatchNorm2d(64),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(outplanes // 2, 64, 1),
        #     nn.Conv2d(64, 64, 3, padding=3, dilation=3, bias=False),
        #     nn.BatchNorm2d(64),
        # )
        self.branch1 = nn.Sequential(
            BasicConv2d((outplanes //3) +1, inchannel, 1),
            BasicConv2d(inchannel, inchannel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(inchannel, inchannel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(inchannel, inchannel, 3, padding=3, dilation=3),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(True)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(outplanes //3+1, inchannel, 1),
            BasicConv2d(inchannel, inchannel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(inchannel, inchannel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(inchannel, inchannel, 3, padding=5, dilation=5),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(True)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(outplanes //3-1, inchannel, 1),
            BasicConv2d(inchannel, inchannel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(inchannel, inchannel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(inchannel, inchannel, 3, padding=7, dilation=7),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(True)
        )

        # 1*1的卷积层
        self.outconv = nn.Sequential(
            nn.Conv2d(384, 64, 1, 1, 0), #  nn.Conv2d(inchannel * 2
            nn.BatchNorm2d(64),
        )
        self.outconv2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        input = x
        x = self.inconv(x)

        # scales个部分
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        ys.append(F.relu(self.branch1(xs[0])))
        ys.append(F.relu(self.branch2(xs[1]) + ys[0]))
        ys.append(F.relu(self.branch3(xs[2]) + ys[1]))

        y = torch.cat(ys, 1)

        y = self.outconv(y)
        input = self.outconv2(input)
        output = F.relu(y+input)

        return output


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

class SAM(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(SAM, self).__init__()
        self.conv_atten = conv3x3(2, 1)
        self.conv = conv3x3(in_ch, out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        atten = torch.cat([avg_out, max_out], dim=1)
        atten = torch.sigmoid(self.conv_atten(atten))


        out = torch.mul(x, atten)
        out = F.relu(self.bn(self.conv(out))+x, inplace=True)
        return self.sigmoid(out)



class MSCA(nn.Module):
    def __init__(self, channels=64, r=16):
        super(MSCA, self).__init__()
        out_channels = int(channels // r)
        # local att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )

        # global att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)+x
        xg = self.global_att(x)+x
        xlg = xl + xg
        wei = self.sig(xlg)
        return wei

class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v = nn.BatchNorm2d(64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')

        out1h = F.relu(self.bn1h(self.conv1h(left)), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h + left)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down)), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v + down)), inplace=True)
        fuse = out2h * out2v

        out3h = F.relu(self.bn3h(self.conv3h(fuse)), inplace=True) + out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h + fuse)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse)), inplace=True) + out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v + fuse)), inplace=True)

        return out4h, out4v

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cfm45 = CFM()
        self.cfm34 = CFM()
        self.cfm23 = CFM()

    def forward(self, out2h, out3h, out4h, out5v, fback=None):
        if fback is not None:
            refine5 = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4 = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3 = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2 = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            out5v = out5v + refine5
            out4h, out4v = self.cfm45(out4h + refine4, out5v)
            out3h, out3v = self.cfm34(out3h + refine3, out4v)
            out2h, pred = self.cfm23(out2h + refine2, out3v)
        else:
            out4h, out4v = self.cfm45(out4h, out5v)
            out3h, out3v = self.cfm34(out3h, out4v)
            out2h, pred = self.cfm23(out2h, out3v)
        return out2h, out3h, out4h, out5v, pred


class SPRANet(nn.Module):
    def __init__(self, cfg):
        super(SPRANet, self).__init__()

        # #resnet
        # self.backbone = ResNet()
        # path = './pvt/resnet50-19c8e357.pth'
        # save_model = torch.load(path)
        # model_dict = self.backbone.state_dict()
        # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # self.backbone.load_state_dict(model_dict)
        #
        # self.squeeze5 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        # self.squeeze4 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        # self.squeeze3 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        # self.squeeze2 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())


        #pvt
        # self.backbone = pvt_v2_b2()
        # path = './pvt/pvt_v2_b2.pth'
        # save_model = torch.load(path)
        # model_dict = self.backbone.state_dict()
        # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # self.backbone.load_state_dict(model_dict)
        #
        # self.squeeze5 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())
        # self.squeeze4 = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())
        # self.squeeze3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())
        # self.squeeze2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())

        # #swin
        self.backbone = SwinTransformer(img_size=384,
                                       embed_dim=128,
                                       depths=[2, 2, 18, 2],
                                       num_heads=[4, 8, 16, 32],
                                       window_size=12)

        pretrained_dict = torch.load('./pvt/swin_base_patch4_window12_384_22k.pth')["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.backbone.state_dict()}
        self.backbone.load_state_dict(pretrained_dict)

        self.squeeze5 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())
        self.squeeze4 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())
        self.squeeze3 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())
        self.squeeze2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())

        self.res = Res2NetBlock(64, 128)
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        self.msca1 = MSCA(64)
        self.att = SAM(64,64)
        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearp2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, shape=None):
        pvt = self.backbone(x)
        # #  #pvt   ResNet
        # out2h = pvt[0]
        # out3h = pvt[1]
        # out4h = pvt[2]
        # out5v = pvt[3]

        #swin
        out2h = pvt[4]
        out3h = pvt[3]
        out4h = pvt[2]
        out5v = pvt[1]




        out2h, out3h, out4h, out5v = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(
            out5v)

        out2h = self.att(out2h)
        out3h = self.att(out3h)
        out4h = self.att(out4h)
        out5v = self.att(out5v)

        out2h = self.res(out2h)
        out3h = self.res(out3h)
        out4h = self.res(out4h)
        out5v = self.res(out5v)


        out2h, out3h, out4h, out5v, pred1 = self.decoder1(out2h, out3h, out4h, out5v)
        out2h, out3h, out4h, out5v, pred2 = self.decoder2(out2h, out3h, out4h, out5v, pred1)

        pred1 = self.msca1(pred1)
        pred2 = self.msca1(pred2)
        out2h = self.msca1(out2h)
        out3h = self.msca1(out3h)
        out4h = self.msca1(out4h)
        out5v = self.msca1(out5v)

        shape = x.size()[2:] if shape is None else shape
        pred1 = F.interpolate(self.linearp1(pred1), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.linearp2(pred2), size=shape, mode='bilinear')

        out2h = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear')
        out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear')
        out4h = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear')
        out5h = F.interpolate(self.linearr5(out5v), size=shape, mode='bilinear')

        return pred1, pred2, out2h, out3h, out4h, out5h

