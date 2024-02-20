import torch


LRELU_SLOPE = 0.1


def init_weights(m: torch.nn.Module, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation=1):
    return (kernel_size * dilation - dilation) >> 1
