import torch


class SSConv2d(torch.nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=None, groups=1, dilation=1, alpha=0.04):
        super(SSConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias)
        self.alpha, self.groups = alpha, groups

    def create_shuffle_indices(self, x):
        _, in_planes, height, width = x.size()
        self.shuffle_until_here = int(in_planes * self.alpha)
        # if self.shuffle_until_here = 0, then it's exactly same as regular convolution
        if self.shuffle_until_here >= 1:
            self.register_buffer('random_indices', torch.randperm(self.shuffle_until_here * height * width))

    @staticmethod
    def _group(shuffled_x, non_shuffled_x):
        batch, ch_ns, height, width = non_shuffled_x.shape
        _, ch_s, _, _ = shuffled_x.shape
        length = int(ch_ns / ch_s)
        residue = ch_ns - length * ch_s
        # shuffled_x is interleaved
        if residue == 0:
            return torch.cat((shuffled_x.unsqueeze(1), non_shuffled_x.view(batch, length, ch_s, height, width)), 1).view(batch, ch_ns + ch_s, height, width)
        else:
            return torch.cat((torch.cat((shuffled_x.unsqueeze(1), non_shuffled_x[:, residue:].view(batch, length, ch_s, height, width)), 1).view(batch, ch_ns + ch_s - residue, height, width), non_shuffled_x[:, :residue]), 1)

    def shuffle(self, x):
        if self.shuffle_until_here >= 1:
            # ss convolution
            shuffled_x, non_shuffled_x = x[:, :self.shuffle_until_here], x[:, self.shuffle_until_here:]
            batch, ch, height, width = shuffled_x.size()
            shuffled_x = torch.index_select(shuffled_x.view(batch, -1), 1, self.random_indices).view(batch, ch, height, width)
            if self.groups >= 2:
                return self._group(shuffled_x, non_shuffled_x)
            else:
                return torch.cat((shuffled_x, non_shuffled_x), 1)
        else:
            # regular convolution
            return x

    def forward(self, x):
        if hasattr(self, 'random_indices') is False:
            # create random permutation matrix at initialization
            self.create_shuffle_indices(x)
        # spatial shuffling
        x = self.shuffle(x)
        # regular convolution
        x = self.conv(x)
        return x


if __name__ == '__main__':
    from torch.autograd import Variable
    x = Variable(torch.randn(2, 100, 32, 32))
    net = SSConv2d(100, 100, 3, 1, 1, alpha=0.04)
    y = net(x)
