import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import download_file_from_gdrive

# mapping network
#################

def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(torch.pow(x, 2), dim=1, keepdim=True) + epsilon)

class MappingNetwork(nn.Module):
    def __init__(self,
                 num_layers=8,
                 latent_size=512, # input size
                 mapping_size=512, # hidden layer size
                 dlatent_size=512, # disentangled latent size (output)
                 dlatent_broadcast=18, # how many dlatent copies in output
                 normalize_latents=True,
                ):
        super(MappingNetwork, self).__init__()
        self.dlatent_broadcast = dlatent_broadcast
        self.normalize_latents = normalize_latents

        self.net = self._make_layers(num_layers, latent_size, mapping_size, dlatent_size)

    def _make_layers(self, num_layers, latent_size, mapping_size, dlatent_size):
        layers = []
        for layer_idx in range(num_layers):
            in_size = latent_size if layer_idx == 0 else mapping_size
            out_size = dlatent_size if layer_idx == num_layers - 1 else mapping_size
            layers += [nn.Linear(in_size, out_size), nn.LeakyReLU(negative_slope=0.2)]
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.normalize_latents:
            x = pixel_norm(x)

        x = self.net(x)

        if self.dlatent_broadcast is not None:
            x = x[None, :, :] # add a dimension
            x = x.repeat(1, self.dlatent_broadcast, 1)

        return x

# synth network
###############

def instance_norm(x, epsilon=1e-8):
    assert len(x.shape) == 4 # NCHW
    x -= torch.mean(x, dim=[2, 3], keepdim=True)
    x = x * torch.rsqrt(torch.mean(torch.pow(x, 2), dim=[2, 3], keepdim=True) + epsilon)
    return x

def upscale2d(x, factor=2):
    assert len(x.shape) == 4
    assert isinstance(factor, int) and factor >= 1

    # early exit
    if factor==1:
        return x

    s = x.shape
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = x.view(-1, s[1], s[2] * factor, s[3] * factor)
    return x

def blur2d(x, f=[1, 2, 1], normalize=True, flip=False, stride=1):
    assert len(x.shape) == 4

    # modify kernel
    f = np.array(f, dtype=np.float32)
    if len(f.shape) == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert len(f.shape) == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[np.newaxis, np.newaxis, :, :]
    num_channels = x.shape[1]

    f = torch.from_numpy(f)
    f = f.repeat(num_channels, 1, 1, 1)

    # convolve via depthwise_conv
    x = F.conv2d(x, f, groups=num_channels, padding=1)
    return x

class Blur2D(nn.Module):
    def __init__(self, num_channels, f=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(Blur2D, self).__init__()

        self.num_channels = num_channels
        f = np.array(f, dtype=np.float32)
        if len(f.shape) == 1:
            f = f[:, np.newaxis] * f[np.newaxis, :]
        assert len(f.shape) == 2
        if normalize:
            f /= np.sum(f)
        if flip:
            f = f[::-1, ::-1]
        f = f[np.newaxis, np.newaxis, :, :]

        f = torch.from_numpy(f)
        f = f.repeat(num_channels, 1, 1, 1)
        #self.f = f
        self.register_buffer("f", f)

    def forward(self, x):
        x = F.conv2d(x, self.f, groups=self.num_channels, padding=1)
        return x

class StyleMod(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(StyleMod, self).__init__()
        self.dense = nn.Linear(channels_in, channels_out)

    def forward(self, x, dlatent):
        style = self.dense(dlatent)
        shape = [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2)
        style = style.view(*shape)
        return x * (style[:, 0] + 1) + style[:, 1]

class LearnableBias(nn.Module):
    def __init__(self, num_channels):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        if len(x.shape) == 2:
            return x + self.bias
        return x + self.bias.view(1, -1, 1, 1)

class NoiseBlock(nn.Module):
    def __init__(self, num_channels, height, width):
        # num_channels - input tensor's channels
        # height - noise tensor's height
        # width - noise tensor's width
        super(NoiseBlock, self).__init__()
        self.num_channels = num_channels
        self.noise = nn.Parameter(torch.randn(1, 1, height, width))
        self.weight = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, randomize_noise):
        assert len(x.shape) == 4 # NCHW
        assert x.shape[1] == self.num_channels
        if randomize_noise:
            noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]))
        else:
            noise = self.noise
        return x + noise * self.weight.view(1, -1, 1, 1) # nchw + (n/1)1hw * 1c11 = nchw + (n1/1)chw

class LayerEpilogue(nn.Module):
    def __init__(self, dlatent_channels, input_channels,
                 noise_height, noise_width,
                 use_noise=True, use_pixel_norm=False,
                 use_instance_norm=True, use_style=True):
        super(LayerEpilogue, self).__init__()
        self.use_noise = use_noise
        self.use_pixel_norm = use_pixel_norm
        self.use_instance_norm = use_instance_norm
        self.use_style = use_style
        self.noise = NoiseBlock(input_channels, noise_height, noise_width)
        self.bias = LearnableBias(input_channels)
        self.style = StyleMod(dlatent_channels, 2*input_channels)

    def forward(self, x, dlatent, randomize_noise=True):
        if self.use_noise:
            x = self.noise(x, randomize_noise)
        x = self.bias(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        if self.use_pixel_norm:
            x = pixel_norm(x)
        if self.use_instance_norm:
            x = instance_norm(x)
        if self.use_style:
            x = self.style(x, dlatent)
        return x

class ConstSynthBlock(nn.Module):
    def __init__(self, dlatent_channels, input_channels, height=4, width=4):
        super(ConstSynthBlock, self).__init__()
        self.const = nn.Parameter(torch.ones(1, input_channels, height, width))
        self.const_epilogue = LayerEpilogue(dlatent_channels, input_channels, height, width)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)
        self.conv_epilogue = LayerEpilogue(dlatent_channels, input_channels, height, width)

    def forward(self, dlatent, randomize_noise):
        assert dlatent.shape[1] == 2
        x = self.const
        x = self.const_epilogue(x, dlatent[:, 0], randomize_noise)
        s0 = x.shape
        x = self.conv(x)
        s1 = x.shape
        assert s0 == s1
        x = self.conv_epilogue(x, dlatent[:, 1], randomize_noise)
        return x

class UpscaleConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, fused_scale="auto"):
        super(UpscaleConv2d, self).__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1
        assert fused_scale in [True, False, "auto"]
        self.fused_scale = fused_scale

        self.weight = nn.Parameter(torch.randn(output_channels, input_channels, kernel_size, kernel_size))

    def forward(self, x):
        fused_scale = self.fused_scale
        if fused_scale=="auto":
            fused_scale = min(x.shape[2:]) * 2 >= 128

        if not fused_scale:
            x = upscale2d(x)
            x = F.conv2d(x, self.weight, padding=1)
        else:
            w = self.weight.permute(1, 0, 2, 3)
            w = F.pad(w, (1,1,1,1))
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
        return x

class SynthBlock(nn.Module):
    def __init__(self, dlatent_channels, input_channels, output_channels, height, width):
        super(SynthBlock, self).__init__()

        self.conv0 = UpscaleConv2d(input_channels, output_channels, kernel_size=3)
        self.blur2d = Blur2D(output_channels)
        self.conv0_epilogue = LayerEpilogue(dlatent_channels, output_channels, height, width)
        self.conv1 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.conv1_epilogue = LayerEpilogue(dlatent_channels, output_channels, height, width)

    def forward(self, x, dlatent, randomize_noise):
        assert dlatent.shape[1] == 2
        x = self.conv0(x)
        x = self.blur2d(x)
        x = self.conv0_epilogue(x, dlatent[:, 0], randomize_noise)
        x = self.conv1(x)
        x = self.conv1_epilogue(x, dlatent[:, 1], randomize_noise)
        return x


class ToRGBBlock(nn.Module):
    def __init__(self, input_channels):
        super(ToRGBBlock, self).__init__()

        self.conv = nn.Conv2d(input_channels, 3, kernel_size=1, bias=False)
        self.bias = LearnableBias(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bias(x)
        return x


class SynthNetwork(nn.Module):
    def __init__(self, dlatent_channels=512, resolution=1024):
        super(SynthNetwork, self).__init__()

        def nf(stage, fmap_base = 8192, fmap_decay = 1.0, fmap_max = 512):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.resolution = resolution

        # early layer
        self.block_4x4 = ConstSynthBlock(dlatent_channels, nf(1), 4, 4)

        # middle layers
        res_log2 = int(np.log2(resolution))
        for res in range(3, res_log2 + 1):
            hw = 2**res
            setattr(self, "block_{0}x{0}".format(hw), SynthBlock(dlatent_channels, nf(res - 2), nf(res - 1), hw, hw))

        # rgb image branch
        #self.torgb = ToRGBBlock(nf(res - 1))
        self.torgb = nn.Conv2d(nf(res - 1), 3, kernel_size=1)

    def forward(self, dlatents, use_noise=False):
        x = self.block_4x4(dlatents[:, :2], use_noise)
        if torch.isnan(x).any():
            print("Nan detected!")
            import pdb; pdb.set_trace()

        res_log2 = int(np.log2(self.resolution))
        for res in range(3, res_log2 + 1):
            hw = 2**res
            x = getattr(self, "block_{0}x{0}".format(hw))(x, dlatents[:, (res * 2 - 4): (res * 2 - 2)], use_noise)
            if torch.isnan(x).any():
                print("Nan detected!")
                import pdb; pdb.set_trace()

        x = self.torgb(x)
        return x

# stylegan
##########

class StyleGAN(nn.Module):
    WEIGHTS_MAP = {
        "ffhq": {
            "fname":"stylegan_ffhq.pt",
            "url": "https://drive.google.com/uc?id=1qnG4jFWnXh3WYqBG4fLw7hHgYdpzng51",
        }
    }

    def __init__(self, weights=None):
        super(StyleGAN, self).__init__()
        self.mapping = MappingNetwork()
        self.synthesis = SynthNetwork()

        if weights is not None:
            self._load_weights(weights)

    def _load_weights(self, weights):
        weights_data = self.WEIGHTS_MAP.get(weights)
        fname = weights_data.get("fname")
        if fname is None:
            raise RuntimeError("No such weights: {}".format(fname))
        path = os.path.expanduser(os.path.join("~", "neural_obfuscator", fname))
        if not os.path.exists(path):
            download_file_from_gdrive(weights_data["url"], path)
        self.load_state_dict(torch.load(path), strict=False)

    def forward(self, latents, use_noise=False, postprocess=True):
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents)
        x = self.mapping(latents)
        x = self.synthesis(x, use_noise=use_noise)
        if postprocess:
            x = self.postprocess(x)
        return x

    @staticmethod
    def postprocess(images, drange=[-1, 1]):
        scale = 255. / (drange[1] - drange[0])
        images = images * scale + (0.5 - drange[0] * scale)
        images = images.clamp(0, 255)
        images = images.data.numpy().astype("uint8")
        images = images.transpose(0, 2, 3, 1) # NHWC
        images = images[:, :, :, ::-1] # bgr
        return images
