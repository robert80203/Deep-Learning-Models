import torch
import torch.nn as nn
import torch.nn.functional as F

class downblock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attn=False):
        super(downblock, self).__init__()
        self.use_attn = use_attn
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=1,padding=1,bias=False),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=False),
            nn.LeakyReLU(0.2)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1,padding=0,bias=False),
            nn.LeakyReLU(0.2)
        )
        self.downsampler = nn.AvgPool2d(kernel_size=2, stride=2)
        if use_attn:
            self.attn = SelfAttention(out_ch)
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.downsampler(x)
        x = self.conv2(x)
        if self.use_attn:
            x = self.attn(x)
        residual = self.downsampler(residual)
        residual = self.shortcut(residual)
        return x + residual
        
class upblock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attn=False):
        super(upblock, self).__init__()
        self.use_attn = use_attn
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(in_ch)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1,padding=0,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
        )
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
        if use_attn:
            self.attn = SelfAttention(out_ch)
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.upsampler(x)
        x = self.conv2(x)
        if self.use_attn:
            x = self.attn(x)
        residual = self.upsampler(residual)
        residual = self.shortcut(residual)
        return x + residual
        
class SelfAttention(nn.Module):
    """Self Attention Module as in https://arxiv.org/pdf/1805.08318.pdf
    """
    def __init__(self, C):
        """
        Args:
            C: Number of channels in the input feature maps torch attend over.
        """
        super(SelfAttention, self).__init__()

        self.f_x = nn.Conv2d(in_channels=C, out_channels=C // 8, kernel_size=1)
        self.g_x = nn.Conv2d(in_channels=C, out_channels=C // 8, kernel_size=1)
        self.h_x = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        """Projects input x to three different features spaces using f(x),
        g(x) and h(x). Applying outer product to outputs of f(x) and g(x).
        A softmax is applied on top to get a attention map. The attention map
        is applied to the output of h(x) to get the final attended features.
        Args:
            x: input features maps. shape=(B,C,H,W).
        Returns:
            y: Attended features of shape=(B,C,H,W).
        """
        B, C, H, W = x.size()
        N = H * W
        
        f_w = self.f_x(x)
        f = f_w.view(B, C // 8, N)

        g = self.g_x(x).view(B, C // 8, N)
    
        ###################################
        #inner product
        ###################################
        s = torch.bmm(f.permute(0, 2, 1), g)  # f(x)^{T} * g(x)
        beta = F.softmax(s, dim=1)

        h = self.h_x(x).view(B, C, N)
        o = torch.bmm(h, beta).view(B, C, H, W)

        y = self.gamma * o + x

        return y