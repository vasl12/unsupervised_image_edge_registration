import math
import os
import torch
from torch import nn as nn
from torch.nn import functional as F

from utils.misc import param_ndim_setup

def convNd(ndim,
           in_channels,
           out_channels,
           kernel_size=3,
           stride=1,
           padding=1,
           a=0.):
    """
    Convolution of generic dimension
    Args:
        in_channels: (int) number of input channels
        out_channels: (int) number of output channels
        kernel_size: (int) size of the convolution kernel
        stride: (int) convolution stride (step size)
        padding: (int) outer padding
        ndim: (int) model dimension
        a: (float) leaky-relu negative slope for He initialisation

    Returns:
        (nn.Module instance) Instance of convolution module of the specified dimension
    """
    conv_nd = getattr(nn, f"Conv{ndim}d")(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding)
    nn.init.kaiming_uniform_(conv_nd.weight, a=a)
    return conv_nd


def interpolate_(x, scale_factor=None, size=None, mode=None):
    """ Wrapper for torch.nn.functional.interpolate """
    if mode == 'nearest':
        mode = mode
    else:
        ndim = x.ndim - 2
        if ndim == 1:
            mode = 'linear'
        elif ndim == 2:
            mode = 'bilinear'
        elif ndim == 3:
            mode = 'trilinear'
        else:
            raise ValueError(f'Data dimension ({ndim}) must be 2 or 3')
    y = F.interpolate(x,
                      scale_factor=scale_factor,
                      size=size,
                      mode=mode,
                      )
    return y


class UNet(nn.Module):
    """
    Adpated from the U-net used in VoxelMorph:
    https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """
    def __init__(self,
                 ndim,
                 enc_channels=(16, 32, 32, 32, 32),
                 dec_channels=(32, 32, 32, 32),
                 out_channels=(16, 16),
                 conv_before_out=True,
                 use_edges=False    # TODO: update config file so we can change to edges and no edges
                 ):
        super(UNet, self).__init__()

        self.ndim = ndim
        self.use_edges = use_edges

        # encoder layers
        self.enc = nn.ModuleList()
        for i in range(len(enc_channels)):
            # in_ch = 2 when we feed the network with cat tar, src
            # in_ch = 1 when we use 2 encoders with shared weights
            in_ch = 2 if i == 0 else enc_channels[i - 1]
            stride = 1 if i == 0 else 2
            self.enc.append(
                nn.Sequential(
                    convNd(ndim, in_ch, enc_channels[i], stride=stride, a=0.2),
                    # nn.BatchNorm2d(enc_channels[i]),
                    nn.LeakyReLU(0.2)
                )
            )
        if self.use_edges:
            # encoder layers for the help images
            self.enc_edge = nn.ModuleList()
            for i in range(len(enc_channels)):
                # in_ch = 2 when we feed the network with cat tar, src
                # in_ch = 1 when we use 2 encoders with shared weights
                in_ch = 2 if i == 0 else enc_channels[i - 1]
                stride = 1 if i == 0 else 2
                self.enc_edge.append(
                    nn.Sequential(
                        convNd(ndim, in_ch, enc_channels[i], stride=stride, a=0.2),
                        nn.LeakyReLU(0.2)
                    )
                )

        # decoder layers
        self.dec = nn.ModuleList()
        for i in range(len(dec_channels)):
            # in_ch = enc_channels[-1] when we feed the encoder with the concat src, tar data
            # in_ch = enc_channels[-1]*2 when we concatenate the features after the encoder
            in_ch = enc_channels[-1] if i == 0 else dec_channels[i-1] + enc_channels[-i-1]
            if self.use_edges:
                in_ch = 2*enc_channels[-1] if i == 0 else dec_channels[i - 1] + 2*enc_channels[-i - 1]
                # in_ch = 2*enc_channels[-1] if i == 0 else dec_channels[i - 1] + enc_channels[-i - 1]
            self.dec.append(
                nn.Sequential(
                    convNd(ndim, in_ch, dec_channels[i], a=0.2),
                    nn.LeakyReLU(0.2)
                )
            )


        # decoder layers
        self.dec_seg = nn.ModuleList()
        for i in range(len(dec_channels)):
            # in_ch = enc_channels[-1] when we feed the encoder with the concat src, tar data
            # in_ch = enc_channels[-1]*2 when we concatenate the features after the encoder
            in_ch = enc_channels[-1] if i == 0 else dec_channels[i - 1] + enc_channels[-i - 1]
            if self.use_edges:
                in_ch = 2 * enc_channels[-1] if i == 0 else dec_channels[i - 1] + 2 * enc_channels[-i - 1]
            self.dec_seg.append(
                nn.Sequential(
                    convNd(ndim, in_ch, dec_channels[i], a=0.2),
                    nn.LeakyReLU(0.2)
                )
            )
        # out is num of classes for segmentation task
        if use_edges:
            self.dec_seg_out = convNd(ndim, 64, 6, kernel_size=1, padding=0)
        else:
            self.dec_seg_out = convNd(ndim, 48, 6, kernel_size=1, padding=0)


        # upsampler
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # (optional) conv layers before prediction
        if conv_before_out:
            self.out_layers = nn.ModuleList()
            for i in range(len(out_channels)):
                in_ch = dec_channels[-1] + enc_channels[0] if i == 0 else out_channels[i-1]
                if self.use_edges:
                    in_ch = dec_channels[-1] + 2*enc_channels[0] if i == 0 else out_channels[i - 1]
                self.out_layers.append(
                    nn.Sequential(
                        convNd(ndim, in_ch, out_channels[i], a=0.2),  # stride=1
                        nn.LeakyReLU(0.2)
                    )
                )

            # final prediction layer with additional conv layers
            self.out_layers.append(
                convNd(ndim, out_channels[-1], ndim)
            )

        else:

            # final prediction layer without additional conv layers
            self.out_layers = nn.ModuleList()
            self.out_layers.append(
                convNd(ndim, dec_channels[-1] + enc_channels[0], ndim)
            )

    def forward(self, batch):
        tar = batch['target']
        src = batch['source']

        x = torch.cat((tar, src), dim=1)
        # encoder for target image
        fm_enc1 = [x]
        for enc in self.enc:
            fm_enc1.append(enc(fm_enc1[-1]))

        if self.use_edges:
            tar_edges = batch['target_edges']
            src_edges = batch['source_edges']

            x = torch.cat((tar_edges, src_edges), dim=1)
            # encoder for source image
            fm_enc2 = [x]
            for enc_edge in self.enc:
                fm_enc2.append(enc_edge(fm_enc2[-1]))



            # concatenate features after the encoder
            dec_out = torch.cat((fm_enc1[-1], fm_enc2[-1]), dim=1)
        else:
            # decoder: conv + upsample + concatenate skip-connections (to full resolution)
            dec_out = fm_enc1[-1]

        for i, dec in enumerate(self.dec):
            dec_out = dec(dec_out)
            dec_out = self.upsample(dec_out)

            if self.use_edges:
                dec_out = torch.cat([dec_out, fm_enc1[-2 - i], fm_enc2[-2 - i]], dim=1)
            else:
                dec_out = torch.cat([dec_out, fm_enc1[-2 - i]], dim=1)

        # further convs and prediction
        y = dec_out
        for out_layer in self.out_layers:
            y = out_layer(y)
        return y


class CubicBSplineNet(UNet):
    def __init__(self,
                 ndim,
                 enc_channels=(16, 32, 32, 32, 32),
                 dec_channels=(32, 32, 32, 32),
                 resize_channels=(32, 32),
                 cps=(5, 5, 5),
                 img_size=(176, 192, 176),
                 use_edges=False
                 ):
        """
        Network to parameterise Cubic B-spline transformation
        """
        super(CubicBSplineNet, self).__init__(ndim=ndim,
                                              enc_channels=enc_channels,
                                              conv_before_out=False,
                                              use_edges=use_edges,
                                              )

        self.use_edges = use_edges


        # determine and set output control point sizes from image size and control point spacing
        img_size = param_ndim_setup(img_size, ndim)
        cps = param_ndim_setup(cps, ndim)
        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")
        self.output_size = tuple([int(math.ceil((imsz-1) / c) + 1 + 2)
                                  for imsz, c in zip(img_size, cps)])

        # Network:
        # encoder: same u-net encoder
        # decoder: number of decoder layers / times of upsampling by 2 is decided by cps
        num_dec_layers = 4 - int(math.ceil(math.log2(min(cps))))
        self.dec = self.dec[:num_dec_layers]

        # conv layers following resizing
        self.resize_conv = nn.ModuleList()
        for i in range(len(resize_channels)):
            if i == 0:
                if num_dec_layers > 0:
                    if self.use_edges:
                        in_ch = dec_channels[num_dec_layers - 1] + 2*enc_channels[-num_dec_layers]
                        # in_ch = dec_channels[num_dec_layers - 1] + enc_channels[-num_dec_layers]
                    else:
                        in_ch = dec_channels[num_dec_layers-1] + enc_channels[-num_dec_layers]
                else:
                    if self.use_edges:
                        in_ch = 2 * enc_channels[-1]
                    else:
                        in_ch = enc_channels[-1]
            else:
                in_ch = resize_channels[i-1]
            out_ch = resize_channels[i]
            self.resize_conv.append(nn.Sequential(convNd(ndim, in_ch, out_ch, a=0.2),
                                                  nn.LeakyReLU(0.2)))

        # final prediction layer
        delattr(self, 'out_layers')  # remove u-net output layers
        self.out_layer = convNd(ndim, resize_channels[-1], ndim)

    def forward(self, batch):
        tar = batch['target']
        src = batch['source']

        x = torch.cat((tar, src), dim=1)
        # encoder for target image
        fm_enc1 = [x]
        for enc in self.enc:
            fm_enc1.append(enc(fm_enc1[-1]))

        if self.use_edges:
            tar_edges = batch['target_edges']
            src_edges = batch['source_edges']

            # x = torch.cat((tar_edges, src_edges, tar_edges_theta, tar_edges_fi, src_edges_theta, src_edges_fi), dim=1)
            x = torch.cat((tar_edges, src_edges), dim=1)
            # encoder for source image
            fm_enc2 = [x]
            for enc_edge in self.enc:
                fm_enc2.append(enc_edge(fm_enc2[-1]))

            # concatenate features after the encoder
            dec_out = torch.cat((fm_enc1[-1], fm_enc2[-1]), dim=1)
        else:
            dec_out = fm_enc1[-1]

        # decoder: conv + upsample + concatenate skip-connections
        if len(self.dec) > 0:
            # dec_out = cat_enc
            for i, dec in enumerate(self.dec):
                dec_out = dec(dec_out)
                dec_out = self.upsample(dec_out)
                if self.use_edges:
                    dec_out = torch.cat([dec_out, fm_enc1[-2-i], fm_enc2[-2-i]], dim=1)
                    # dec_out = torch.cat([dec_out, fm_enc1[-2-i]], dim=1)
                else:
                    dec_out = torch.cat([dec_out, fm_enc1[-2 - i]], dim=1)
        else:
            dec_out = fm_enc1

        # resize output of encoder-decoder
        x = interpolate_(dec_out, size=self.output_size)

        # layers after resize
        for resize_layer in self.resize_conv:
            x = resize_layer(x)
        y = self.out_layer(x)

        return y