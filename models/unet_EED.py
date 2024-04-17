import torch
import torch.nn as nn
import torch.nn.functional as F
from models import EED
import configs.config_setting as cfg
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet_EED(nn.Module):
    def __init__(self, activefunc,droprate,kernel_size,n_channels, n_classes, bilinear=True):
        super(UNet_EED, self).__init__()

        if activefunc == 'relu':
            self.act = nn.ReLU()
        elif activefunc == 'gelu':
            self.act = nn.GELU()
        elif activefunc == 'tanh':
            self.act = nn.Tanh()
        self.drop = nn.Dropout(p=droprate)  #
        self.ker = kernel_size  #
        self.pad = (self.ker - 1) // 2  #

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = EED.CR_B(in_channel=512,out_channel=cfg.y0_channel,kernel_size=self.ker,stride=1,padding=self.pad)
        self.up2 = EED.CR_B(in_channel=512,out_channel=cfg.y0_channel,kernel_size=self.ker,stride=1,padding=self.pad)
        self.up3 = EED.CR_B(in_channel=256,out_channel=cfg.y0_channel,kernel_size=self.ker,stride=1,padding=self.pad)
        self.up4 = EED.CR_B(in_channel=128,out_channel=cfg.y0_channel,kernel_size=self.ker,stride=1,padding=self.pad)
        self.up5 = EED.CR_B(in_channel=64, out_channel=cfg.y0_channel,kernel_size=self.ker,stride=1,padding=self.pad)
        self.outc = OutConv(cfg.y0_channel, n_classes)

    def forward(self, x):
        if (cfg.y0_para):
            y0 = nn.Parameter(torch.zeros(x.shape[0], cfg.y0_channel, x.shape[2], x.shape[3])).cuda()
        else:
            y0 = torch.zeros(x.shape[0], cfg.y0_channel, x.shape[2], x.shape[3]).cuda()#255*torch.rand(x.shape[0], cfg.y0_channel, x.shape[2], x.shape[3]).cuda()#x#
        if (cfg.input_weight):
            input_weight = nn.Parameter(torch.ones(5, 2)).cuda()
        else:
            input_weight = torch.ones(5, 2).cuda()

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y5 = self.act(self.up1(x5,y0,1/5,input_weight[0][0],input_weight[0][1],16))
        y5 = self.drop(y5)

        y4 = self.act(self.up2(x4,y5,1/5,input_weight[1][0],input_weight[1][1],8))
        y4 = self.drop(y4)

        y3 = self.act(self.up3(x3,y4,1/5,input_weight[2][0],input_weight[2][1],4))
        y3 = self.drop(y3)

        y2 = self.act(self.up4(x2,y3,1/5,input_weight[3][0],input_weight[3][1],2))
        y2 = self.drop(y2)

        y1 = self.up5(x1, y2, 1/5,input_weight[4][0], input_weight[4][1],1)
        y1 = self.drop(y1)

        y = self.outc(y1)

        return  torch.sigmoid(y)

if __name__ == '__main__':
    net = UNet_EED(activefunc='relu',droprate=0.1,kernel_size=3,n_channels=3, n_classes=1).cuda()
    from thop import profile

    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(net, (dummy_input,))
    print('flops: %.2f M, params: %.2f k' % (flops / 1000000, params / 1000))
    print('net total parameters:', sum(param.numel() for param in net.parameters()))
