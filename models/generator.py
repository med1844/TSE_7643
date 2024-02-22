# copied from https://github.com/daniilrobnikov/vits2/blob/main/model/decoder.py
# basically HiFi-GAN generator
# added type annotation, copied dependency from other files
# modified a little bit to make code analyzer happy

from typing import Collection, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from dataclasses import dataclass
from serde import serde

from .utils import init_weights, get_padding, LRELU_SLOPE


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask=None) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask=None) -> torch.Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


@serde
@dataclass
class GeneratorArgs:
    initial_channel: int
    resblock: type[ResBlock1] | type[ResBlock2]
    resblock_kernel_sizes: Collection[int]
    resblock_dilation_sizes: Collection[Collection[int]]
    upsample_rates: Collection[int]
    upsample_initial_channel: int
    upsample_kernel_sizes: Collection[int]
    gin_channels = 0


class Generator(nn.Module):
    def __init__(
        self,
        args: GeneratorArgs,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(args.resblock_kernel_sizes)
        self.num_upsamples = len(args.upsample_rates)
        self.conv_pre = nn.Conv1d(
            args.initial_channel,
            args.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(args.upsample_rates, args.upsample_kernel_sizes)
        ):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        args.upsample_initial_channel >> i,
                        args.upsample_initial_channel >> (i + 1),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = args.upsample_initial_channel >> (i + 1)
            for j, (k, d) in enumerate(
                zip(args.resblock_kernel_sizes, args.resblock_dilation_sizes)
            ):
                self.resblocks.append(args.resblock(ch, k, d))

        ch = args.upsample_initial_channel // (1 << len(self.ups))
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if args.gin_channels != 0:
            self.cond = nn.Linear(args.gin_channels, args.upsample_initial_channel)

    def forward(
        self, x: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g.mT).mT

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = torch.zeros_like(x)
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
