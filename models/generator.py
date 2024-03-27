# copied from
# - https://github.com/daniilrobnikov/vits2/blob/main/model/decoder.py
# - https://github.com/openvpi/DiffSinger/blob/main/modules/nsf_hifigan/models.py#L42
# basically HiFi-GAN generator
# added type annotation, copied dependency from other files
# modified a little bit to make code analyzer happy

from typing import List, Literal, Optional
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm
from dataclasses import dataclass
from traits import SerdeJson, Json
import numpy as np
from functools import reduce
from operator import mul

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = nn.functional.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = nn.functional.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c in self.convs:
            xt = nn.functional.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


@dataclass
class GeneratorArgs(SerdeJson):
    num_mels: int
    resblock: Literal[1, 2]
    resblock_kernel_sizes: List[int]
    resblock_dilation_sizes: List[List[int]]
    upsample_rates: List[int]
    upsample_initial_channel: int
    upsample_kernel_sizes: List[int]
    sample_rate: int

    def to_json(self) -> Json:
        return {
            "num_mels": self.num_mels,
            "resblock": self.resblock,
            "resblock_kernel_sizes": self.resblock_kernel_sizes,
            "resblock_dilation_sizes": self.resblock_dilation_sizes,
            "upsample_rates": self.upsample_rates,
            "upsample_num_mels": self.upsample_initial_channel,
            "upsample_kernel_sizes": self.upsample_kernel_sizes,
            "sample_rate": self.sample_rate,
        }

    @classmethod
    def from_json(cls, obj: Json) -> "GeneratorArgs":
        return cls(
            num_mels=obj["num_mels"],
            resblock=obj["resblock"],
            resblock_kernel_sizes=obj["resblock_kernel_sizes"],
            resblock_dilation_sizes=obj["resblock_dilation_sizes"],
            upsample_rates=obj["upsample_rates"],
            upsample_initial_channel=obj["upsample_initial_channel"],
            upsample_kernel_sizes=obj["upsample_kernel_sizes"],
            sample_rate=obj["sample_rate"],
        )

    @classmethod
    def default(cls) -> "GeneratorArgs":
        return cls(
            num_mels=128,
            resblock=1,
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4, 4],
            sample_rate=44100,
        )


class SineGen(nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-waveform (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_threshold: F0 threshold for U/V classification (default 0)
    """

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ) -> None:
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values: torch.Tensor, upp: int) -> torch.Tensor:
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        rad_values = (f0_values / self.sampling_rate).fmod(1.0)
        rand_ini = torch.rand(1, self.dim, device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] += rand_ini
        # is_half = rad_values.dtype is not torch.float32
        tmp_over_one = torch.cumsum(rad_values.double(), 1)
        # if is_half:
        #     tmp_over_one = tmp_over_one.half()
        # else:
        tmp_over_one = tmp_over_one.float()
        tmp_over_one *= upp
        tmp_over_one = nn.functional.interpolate(
            tmp_over_one.transpose(2, 1),
            scale_factor=upp,
            mode="linear",
            align_corners=True,
        ).transpose(2, 1)
        rad_values = nn.functional.interpolate(
            rad_values.transpose(2, 1), scale_factor=upp, mode="nearest"
        ).transpose(2, 1)
        tmp_over_one = tmp_over_one.fmod(1.0)
        diff = nn.functional.conv2d(
            tmp_over_one.unsqueeze(1),
            torch.FloatTensor([[[[-1.0], [1.0]]]]).to(tmp_over_one.device),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
        ).squeeze(1)  # Equivalent to torch.diff, but able to export ONNX
        cumsum_shift = (diff < 0).double()
        cumsum_shift = torch.cat(
            (
                torch.zeros((1, 1, self.dim), dtype=torch.double).to(f0_values.device),
                cumsum_shift,
            ),
            dim=1,
        )
        sines = torch.sin(
            torch.cumsum(rad_values.double() + cumsum_shift, dim=1) * 2 * np.pi
        )
        # if is_half:
        #     sines = sines.half()
        # else:
        sines = sines.float()
        return sines

    @torch.no_grad()
    def forward(self, f0: torch.Tensor, upp: int) -> torch.Tensor:
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0 = f0.unsqueeze(-1)
        fn = torch.multiply(
            f0, torch.arange(1, self.dim + 1, device=f0.device).reshape((1, 1, -1))
        )
        sine_waves = self._f02sine(fn, upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).float()
        uv = nn.functional.interpolate(
            uv.transpose(2, 1), scale_factor=upp, mode="nearest"
        ).transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves


class SourceModuleHnNSF(nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ) -> None:
        super().__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, upp: int) -> torch.Tensor:
        sine_wavs = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


class Generator(nn.Module):
    def __init__(
        self,
        args: GeneratorArgs,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(args.resblock_kernel_sizes)
        self.num_upsamples = len(args.upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=args.sample_rate, harmonic_num=8
        )
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(
            nn.Conv1d(
                args.num_mels,
                args.upsample_initial_channel,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(args.upsample_rates, args.upsample_kernel_sizes)
        ):
            c_cur = args.upsample_initial_channel >> (i + 1)
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        args.upsample_initial_channel >> i,
                        args.upsample_initial_channel >> (i + 1),
                        k,
                        u,
                        padding=(k - u) >> 1,
                    )
                )
            )
            if i + 1 < len(args.upsample_rates):
                stride_f0 = reduce(mul, args.upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 << 1,
                        stride=stride_f0,
                        padding=stride_f0 >> 1,
                    )
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))
        resblock = ResBlock1 if args.resblock == 1 else ResBlock2
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = args.upsample_initial_channel >> (i + 1)
            for j, (k, d) in enumerate(
                zip(args.resblock_kernel_sizes, args.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        ch = args.upsample_initial_channel // (1 << len(self.ups))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = reduce(mul, args.upsample_rates)

    def forward(self, x: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """
        x: (B, num_mels, seq_len)
        y: (B, seq_len)
        returns: (B, new_seq)
        """
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = nn.functional.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x = x + self.noise_convs[i](har_source)
            xs = torch.zeros_like(x)
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        x = x.squeeze(1)
        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
