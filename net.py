import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.elu = nn.ELU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.elu(x)
        x = self.bn(x)
        return x


class KenYaNet(nn.Module):
    def __init__(self):
        super(KenYaNet, self).__init__()
        num = 64
        self.R_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )

        self.G_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )

        self.B_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )

        # self.conv = nn.Conv2d(3, 32, kernel_size=1, padding=0, stride=1)
        # # Encoder
        # self.conv1 = ConvBlock(32, 64)
        # self.conv2 = ConvBlock(64, 128)
        # self.conv3 = ConvBlock(128, 256)
        # self.conv4 = ConvBlock(256, 512)
        # self.conv5 = ConvBlock(512, 1024)
        # # Decoder
        # self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        #
        # self.convup5 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        # self.convup4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.convup3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # self.convup2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.convup1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        #
        # self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        r = x[:, 0, :, :].unsqueeze(1)
        g = x[:, 1, :, :].unsqueeze(1)
        b = x[:, 2, :, :].unsqueeze(1)
        r = self.R_net(r)
        g = self.G_net(g)
        b = self.B_net(b)
        return torch.sigmoid(torch.cat([r, g, b], dim=1))
        # conv = self.conv(x)
        # # Encoder
        # conv1 = self.conv1(conv)  # 64
        # conv2 = self.conv2(conv1)  # 128
        # conv3 = self.conv3(conv2)  # 256
        # conv4 = self.conv4(conv3)  # 512
        # conv5 = self.conv5(conv4)  # 1024
        #
        # up5 = self.upconv5(conv5)  # 512
        # up5 = torch.cat([up5, conv4], dim=1)
        # up5 = self.convup5(up5)
        #
        # up4 = self.upconv4(up5)  # 256
        # up4 = torch.cat([up4, conv3], dim=1)
        # up4 = self.convup4(up4)
        #
        # up3 = self.upconv3(up4)  # 128
        # up3 = torch.cat([up3, conv2], dim=1)
        # up3 = self.convup3(up3)
        #
        # up2 = self.upconv2(up3)  # 64
        # up2 = torch.cat([up2, conv1], dim=1)
        # up2 = self.convup2(up2)
        #
        # up1 = self.upconv1(up2)  # 32
        # up1 = torch.cat([up1, conv], dim=1)
        # up1 = self.convup1(up1)
        #
        # out = self.final_conv(up1)
        # out = torch.add(x, out)


if __name__ == '__main__':
    tensor = torch.randn(1, 3, 512, 512).cuda()
    model = KenYaNet().cuda()
    out = (model(tensor))
    print(out)
