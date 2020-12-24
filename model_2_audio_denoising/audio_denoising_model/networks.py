import torch
import torch.nn as nn
import torch.nn.functional as F


# Functions
##############################################################################
def get_network(config):
    return JointModel(config)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# Networks
##############################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:tuple, dilation:tuple,
                 stride=1,
                 norm_fn='bn',
                 act='relu'):
        super(ConvBlock, self).__init__()
        pad = ((kernel_size[0] - 1) // 2 * dilation[0], (kernel_size[1] - 1) // 2 * dilation[1])
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation, bias=norm_fn is None))
        if norm_fn == 'bn':
            block.append(nn.BatchNorm2d(out_channels))
        if act == 'relu':
            block.append(nn.ReLU())
        elif act == 'prelu':
            block.append(nn.PReLU())
        elif act == 'lrelu':
            block.append(nn.LeakyReLU())
        elif act == 'tanh':
            block.append(nn.Tanh())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class ContextAggNet(nn.Module):
    def __init__(self,
                 kernel_sizes,
                 dilations,
                 freq_bins=256,
                 nf=96):
        super(ContextAggNet, self).__init__()
        self.encoder_x = self.make_enc(kernel_sizes, dilations, nf)
        self.encoder_n = self.make_enc(kernel_sizes, dilations, nf // 2, outf=4)

        self.lstm = nn.LSTM(input_size=8*freq_bins + 4*freq_bins, hidden_size=200, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(400, 600),
                                nn.ReLU(True),
                                nn.Linear(600, 600),
                                nn.ReLU(True),
                                nn.Linear(600, freq_bins * 2),
                                nn.Sigmoid())

    def make_enc(self, kernel_sizes, dilations, nf=96, outf=8):
        encoder_x = []
        for i in range(len(kernel_sizes)):
            if i == 0:
                encoder_x.append(ConvBlock(2, nf, kernel_sizes[i], dilations[i]))
            else:
                encoder_x.append(ConvBlock(nf, nf, kernel_sizes[i], dilations[i]))
        encoder_x.append(ConvBlock(nf, outf, (1, 1), (1, 1)))
        return nn.Sequential(*encoder_x)

    def forward(self, x, n):
        f_x = self.encoder_x(x)
        f_x = f_x.view(f_x.size(0), -1, f_x.size(3)).permute(2, 0, 1)
        f_n = self.encoder_n(n)
        f_n = f_n.view(f_n.size(0), -1, f_n.size(3)).permute(2, 0, 1)
        # if self.training is True:
        self.lstm.flatten_parameters()
        f_x, _ = self.lstm(torch.cat([f_x, f_n], dim=2))
        f_x = f_x.permute(1, 0, 2)
        f_x = self.fc(f_x)
        out = f_x.permute(0, 2, 1).view(f_x.size(0), 2, -1, f_x.size(1))
        # f_n = self.encoder_n(n)
        return out


class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1,
                 norm_fn='bn',
                 act='prelu'):
        super(DownConvBlock, self).__init__()
        pad = (kernel_size - 1) // 2 * dilation
        block = []
        block.append(nn.ReflectionPad2d(pad))
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, bias=norm_fn is None))
        if norm_fn == 'bn':
            block.append(nn.BatchNorm2d(out_channels))
        if act == 'prelu':
            block.append(nn.PReLU())
        elif act == 'lrelu':
            block.append(nn.LeakyReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1,
                 norm_fn='bn',
                 act='prelu',
                 up_mode='upconv'):
        super(UpConvBlock, self).__init__()
        pad = (kernel_size - 1) // 2 * dilation
        block = []
        if up_mode == 'upconv':
            block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, pad, dilation,
                                            bias=norm_fn is None))
        elif up_mode == 'upsample':
            block.append(nn.Upsample(scale_factor=2))
            block.append(nn.ReflectionPad2d(pad))
            block.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, 0, dilation,
                                   bias=norm_fn is None))
        if norm_fn == 'bn':
            block.append(nn.BatchNorm2d(out_channels))
        if act == 'prelu':
            block.append(nn.PReLU())
        elif act == 'lrelu':
            block.append(nn.LeakyReLU())
        elif act == 'tanh':
            block.append(nn.Tanh())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class InpaintNet(nn.Module):
    def __init__(self):
        super(InpaintNet, self).__init__()
        ch1 = 64
        ch2 = 128
        ch3 = 256
        self.down1 = nn.Sequential(
            DownConvBlock(2, ch1, 5, 1),
        )
        self.down2 = nn.Sequential(
            DownConvBlock(ch1, ch2, 5, 2),
            DownConvBlock(ch2, ch2, 5, 1),
        )
        self.down3 = nn.Sequential(
            DownConvBlock(2, ch1, 5, 1),
        )
        self.down4 = nn.Sequential(
            DownConvBlock(ch1, ch2, 5, 2),
            DownConvBlock(ch2, ch2, 5, 1),
        )
        self.mid = nn.Sequential(
            DownConvBlock(ch2 * 2, ch3, 3, 2),
            DownConvBlock(ch3, ch3, 3, 1),
            DownConvBlock(ch3, ch3, 3, 1, dilation=2),
            DownConvBlock(ch3, ch3, 3, 1, dilation=4),
            DownConvBlock(ch3, ch3, 3, 1, dilation=8),
            DownConvBlock(ch3, ch3, 3, 1, dilation=16),
            DownConvBlock(ch3, ch3, 3, 1),
            DownConvBlock(ch3, ch3, 3, 1),
            UpConvBlock(ch3, ch2, 3, 2),
        )
        self.up1 = nn.Sequential(
            DownConvBlock(ch2 * 2, ch2, 3, 1),
            UpConvBlock(ch2, ch1, 3, 2),
        )
        self.up2 = nn.Sequential(
            DownConvBlock(ch1 * 2, ch1, 3, 1),
            DownConvBlock(ch1, 2, 3, 1, norm_fn=None, act=None)
        )

    def forward(self, x, y):
        down1 = self.down1(x)
        down2 = self.down2(down1)

        down3 = self.down3(y)
        down4 = self.down4(down3)
        out = self.mid(torch.cat([down2, down4], dim=1))
        if out.shape != down4.shape:
            out = F.interpolate(out, down4.size()[-2:])
        out = self.up1(torch.cat([out, down4], dim=1))
        if out.shape != down3.shape:
            out = F.interpolate(out, down3.size()[-2:])
        out = self.up2(torch.cat([out, down3], dim=1))
        return out


class JointModel(nn.Module):
    def __init__(self, config):
        super(JointModel, self).__init__()
        self.stage1 = InpaintNet()
        self.stage2 = ContextAggNet(config.kernel_sizes, config.dilations)

    def forward(self, x, n):
        n_pred = self.stage1(n, x)
        out = self.stage2(x, n_pred)
        return n_pred, out


def test():
    kernel_sizes = [(1, 7), (7, 1), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5)]
    dilations    = [(1, 1), (1, 1), (1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
    net = ContextAggNet(kernel_sizes, dilations)
    a = torch.randn((4, 2, 257, 205))
    b = torch.randn((4, 2, 257, 205))
    out = net(a)
    print(net)
    print(out.shape)


if __name__ == '__main__':
    test()
