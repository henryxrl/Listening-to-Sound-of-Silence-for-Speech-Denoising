import torch
import torch.nn as nn
import torch.nn.functional as F


# Functions
##############################################################################
def get_network():
    return AudioVisualNet()


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
class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:tuple, dilation:tuple,
                 stride=1,
                 norm_fn='bn',
                 act='relu'):
        super(Conv2dBlock, self).__init__()
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


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:tuple,
                 stride: tuple,
                 norm_fn='bn',
                 act='relu'):
        super(Conv3dBlock, self).__init__()
        pad = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)
        block = []
        block.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, pad, bias=norm_fn is None))
        if norm_fn == 'bn':
            block.append(nn.BatchNorm3d(out_channels))
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


class AudioVisualNet(nn.Module):
    def __init__(self,
                 freq_bins=256,
                 time_bins=178,
                 nf=96):
        super(AudioVisualNet, self).__init__()

        # video_kernel_sizes = [(5, 7, 7), (5, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (1, 3, 3)]
        # video_strides      = [(1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 3, 3), (1, 3, 3)]
        # self.encoder_video = self.make_video_branch(video_kernel_sizes, video_strides, nf=128, outf=256)

        audio_kernel_sizes = [(1, 7), (7, 1), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5)]
        audio_dilations    = [(1, 1), (1, 1), (1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (1, 1), (2, 2), (4, 4)]
        self.encoder_audio = self.make_audio_branch(audio_kernel_sizes, audio_dilations, nf=48, outf=8)

        self.lstm = nn.LSTM(input_size=8*freq_bins, hidden_size=100, bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear(200, 100),
                                nn.ReLU(True),
                                nn.Linear(100, 1))

        # self.fc1 = nn.Sequential(nn.Linear(200, 100),
        #                         nn.ReLU(True),
        #                         nn.Linear(100, 1),
        #                         nn.ReLU(True))

        # self.fc2 = nn.Sequential(nn.Linear(50 * time_bins, 50),
        #                         nn.ReLU(True),
        #                         nn.Linear(50, 1),
        #                         )

    def make_video_branch(self, kernel_sizes, strides, nf=256, outf=256):
        encoder_x = []
        for i in range(len(kernel_sizes)):
            if i == 0:
                encoder_x.append(Conv3dBlock(3, nf, kernel_sizes[i], strides[i]))
            else:
                encoder_x.append(Conv3dBlock(nf, nf, kernel_sizes[i], strides[i]))
        encoder_x.append(Conv3dBlock(nf, outf, (1, 1, 1), (1, 1, 1)))
        return nn.Sequential(*encoder_x)

    def make_audio_branch(self, kernel_sizes, dilations, nf=96, outf=8):
        encoder_x = []
        for i in range(len(kernel_sizes)):
            if i == 0:
                encoder_x.append(Conv2dBlock(2, nf, kernel_sizes[i], dilations[i]))
            else:
                encoder_x.append(Conv2dBlock(nf, nf, kernel_sizes[i], dilations[i]))
        encoder_x.append(Conv2dBlock(nf, outf, (1, 1), (1, 1)))
        return nn.Sequential(*encoder_x)

    def forward(self, s, v_num_frames=60):
        f_s = self.encoder_audio(s)
        f_s = f_s.view(f_s.size(0), -1, f_s.size(3)) # (B, C1, T1)
        f_s = F.interpolate(f_s, size=v_num_frames) # (B, C2, T1)

        # f_v = self.encoder_video(v)
        # f_v = torch.mean(f_v, dim=(-2, -1)) # (B, C2, T2)
        # # f_v = F.interpolate(f_v, size=f_v.size(2) / 5) # (B, C2, T1)
        # f_s = F.interpolate(f_s, size=f_v.size(2)) # (B, C2, T1)
        # # print(f_s.shape, f_v.shape)

        # merge = torch.cat([f_s, f_v], dim=1)
        merge = f_s
        merge = merge.permute(2, 0, 1)  # (T1, B, C1+C2)

        # if self.training is True:
        #     self.lstm.flatten_parameters()
        self.lstm.flatten_parameters()
        merge, _ = self.lstm(merge)

        merge = merge.permute(1, 0, 2)# (B, T1, C1+C2)
        merge = self.fc1(merge)
        out = merge.squeeze(2)
        # print(merge.shape)
        # out = self.fc2(merge.view(merge.size(0), -1))
        return out


def test():
    net = AudioVisualNet()
    print(net)
    # v = torch.randn((8, 3, 60, 224, 224))
    s = torch.randn((8, 2, 256, 178))
    # out = net(v, s)
    out = net(s)
    print(out.shape)


if __name__ == '__main__':
    test()
