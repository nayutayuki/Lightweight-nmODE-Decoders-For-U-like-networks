import pdb
import torch.nn.functional as F
import torch.nn as nn

class C_RB(nn.Module):
    def __init__(self,in_channel,in_channel_post,out_channel,kernel_size=3,stride=1,padding=0):
        super().__init__()

        self.g = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding)
        self.g_post = nn.Conv2d(in_channels=in_channel_post, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                           padding=padding)

        self.sigma = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x, x_post, y, delta, weight_x, weight_x_post, weight_y, scale, scale_post):
        y_diff = -weight_y * y + self.sigma(
            weight_y * y + F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear'))
        y_next = (1 - delta / 2) * weight_y * y + delta / 2 * (
                (1 - delta) * y_diff + self.sigma(
            weight_y * y + delta * y_diff + F.interpolate(self.g_post(weight_x_post * x_post), scale_factor=scale_post,
                                                          mode='bilinear')))
        return y_next

class C_BR(nn.Module):
    def __init__(self,in_channel,in_channel_post,out_channel,kernel_size=3,stride=1,padding=0):
        super().__init__()

        self.g = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding)
        self.g = nn.Conv2d(in_channels=in_channel_post, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                           padding=padding)

        self.sigma = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x, x_post, y, delta, weight_x, weight_x_post, weight_y, scale, scale_post):
        y_diff = -weight_y * y + self.sigma(
            weight_y * y + F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear'))
        y_next = (1 - delta / 2) * weight_y * y + delta / 2 * (
                (1 - delta) * y_diff + self.sigma(
            weight_y * y + delta * y_diff + F.interpolate(self.g_post(weight_x_post * x_post), scale_factor=scale_post,
                                                          mode='bilinear')))
        return y_next

class CR_B(nn.Module):
    def __init__(self,in_channel,in_channel_post,out_channel,kernel_size=3,stride=1,padding=0):
        super().__init__()

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

        self.g_post = nn.Sequential(
            nn.Conv2d(in_channels=in_channel_post, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

        self.sigma = nn.Sequential(
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x, x_post, y, delta, weight_x, weight_x_post, weight_y, scale, scale_post):
        y_diff = -weight_y * y + self.sigma(
            weight_y * y + F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear'))
        y_next = (1 - delta / 2) * weight_y * y + delta / 2 * (
                (1 - delta) * y_diff + self.sigma(
            weight_y * y + delta * y_diff + F.interpolate(self.g_post(weight_x_post * x_post), scale_factor=scale_post,
                                                          mode='bilinear')))
        return y_next

class CB_R(nn.Module):
    def __init__(self,in_channel,in_channel_post,out_channel,kernel_size=3,stride=1,padding=0):
        super().__init__()

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel)
        )

        self.g_post = nn.Sequential(
            nn.Conv2d(in_channels=in_channel_post, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channel)
        )

        self.sigma = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, x, x_post, y, delta, weight_x, weight_x_post, weight_y, scale, scale_post):
        y_diff = -weight_y * y + self.sigma(
            weight_y * y + F.interpolate(self.g(weight_x * x), scale_factor=scale, mode='bilinear'))
        y_next = (1 - delta / 2) * weight_y * y + delta / 2 * (
                (1 - delta) * y_diff + self.sigma(
            weight_y * y + delta * y_diff + F.interpolate(self.g_post(weight_x_post * x_post), scale_factor=scale_post,
                                                          mode='bilinear')))
        return y_next