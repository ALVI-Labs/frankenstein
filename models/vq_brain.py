import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from vector_quantize_pytorch import ResidualVQ, VectorQuantize

import numpy as np
from einops import rearrange

from vector_quantize_pytorch import FSQ




"""
config = dict(C=256, D=64, codebook_size=1024, n_electrodes=512)
model = SoundStream(**config)
count_parameters(model)
x = torch.zeros(16, 768, 512)
loss, pred = model(x)
pred.shape

model.get_quantize_vectors(x)[1].shape"""

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]
    
    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[...,:-self.causal_padding]


class ResidualUnit(nn.Module):
    """
    Input is [batch, emb, time]
    block [ conv -> layer norm -> act -> dropout ]

    To do:
        add res blocks.
    """
    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=3):
        super().__init__()

        # use it instead stride.

        self.conv_dilated = nn.Conv1d(in_channels, in_channels,
                                      kernel_size=kernel_size,
                                      dilation = dilation,
                                      bias=True,
                                      padding='same')

        self.conv1_1 = nn.Conv1d(in_channels, in_channels,
                                 kernel_size=kernel_size,
                                 bias=True,
                                 padding='same')

        self.conv1_2 = nn.Conv1d(in_channels, in_channels,
                                 kernel_size=kernel_size,
                                 bias=True,
                                 padding='same')

        self.conv_final = nn.Conv1d(in_channels, in_channels,
                                    kernel_size=1,
                                    bias=True,
                                    padding='same')
        

    def forward(self, x_input):
        """
        input
            - dilation
            - gated convolution
            - conv final
            - maybe dropout. and LN
        - input + res
        """
        x = self.conv_dilated(x_input)

        flow = torch.tanh(self.conv1_1(x))
        gate = torch.sigmoid(self.conv1_2(x))
        res = flow * gate

        res = self.conv_final(res)

        res = res + x_input
        return res

class ResidualUnitOld(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        
        self.dilation = dilation

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=3, dilation=dilation),
            nn.SiLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=in_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels,
                         dilation=1),
            nn.SiLU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels,
                         dilation=1),
            nn.SiLU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, 
                         dilation=1),
            nn.SiLU(),
            CausalConv1d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=2*stride, 
                         stride=stride)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConvTranspose1d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=2*stride, stride=stride),
            nn.SiLU(),
            ResidualUnit(in_channels=out_channels, 
                         out_channels=out_channels,
                         dilation=1),
            nn.SiLU(),
            ResidualUnit(in_channels=out_channels, 
                         out_channels=out_channels,
                         dilation=1),
            nn.SiLU(),
            ResidualUnit(in_channels=out_channels, 
                         out_channels=out_channels,
                         dilation=1),

        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, C, D, n_electrodes):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=n_electrodes, out_channels=C, kernel_size=3),
            nn.SiLU(),
            EncoderBlock(in_channels=C, out_channels=2*C, stride=1),
            nn.SiLU(),
            EncoderBlock(in_channels=2*C, out_channels=2*C, stride=1),
            nn.SiLU(),
            CausalConv1d(in_channels=2*C, out_channels=D, kernel_size=3)
        )

    def forward(self, x):
        x= rearrange(x, 'b t c -> b c t')
        x= self.layers(x)
        x= rearrange(x, 'b c t -> b t c')
        return x


class Decoder(nn.Module):
    def __init__(self, C, D, n_channels_out):
        super().__init__()
        
        self.layers = nn.Sequential(
            CausalConv1d(in_channels=D, out_channels=2*C, kernel_size=3),
            nn.SiLU(),
            DecoderBlock(in_channels=2*C, out_channels=2*C, stride=1),
            nn.SiLU(),
            DecoderBlock(in_channels=2*C, out_channels=C, stride=1),
            nn.SiLU(),
            CausalConv1d(in_channels=C, out_channels=n_channels_out, kernel_size=3)
        )
    
    def forward(self, x):
        x= rearrange(x, 'b t c -> b c t')
        x= self.layers(x)
        x= rearrange(x, 'b c t -> b t c')
        return x


class SoundStream(nn.Module):
    def __init__(self, C, D, n_electrodes, levels=[8,5,5,5], use_cosine_sim=True):
        super().__init__()
        """
        [1, 16*4, 32] - 1.5 sec 
        encoder
        [1, 16, 8] - 4x downscale + 4x reduce n_channels
        quantize
        [1, 16, 8] - find nearest from codebook. (2048 vectors with 16 size).
        decoder 
        [1, 16*4, 32]
        
        
        vec_emb - [16] * 8 
        codebook - [16] * 2048
        """
        
        self.encoder = Encoder(C=C, D=D, n_electrodes=n_electrodes)
        self.decoder = Decoder(C=C, D=D, n_channels_out=n_electrodes)
        
        self.quantizer = FSQ(levels=levels)

        self.codebook_size = self.quantizer.codebook_size 
        print("self.codebook_size", self.codebook_size)
        
        # self.quantizer = VectorQuantize(
        #                     dim = D,
        #                     codebook_size = codebook_size,
        #                     # codebook_dim = 8,      # paper proposes setting this to 32 or as low as 8 to increase codebook usage
        #                     commitment_weight = 0.25,
        #                     channel_last = True,
        #                     kmeans_init = True,
        #                     threshold_ema_dead_code = 2,
        #                     use_cosine_sim = use_cosine_sim
        #                 )
   
    def forward(self, x, targets=None, date_info=None,  print_loss=False):
        """
        Params:
            x: is tensor with shape (batch, Time, Channels)
        Returns:
            total_loss: mse on nonpadded data
            o: is tensor with shape (batch, Time, Channels)
        """
        e = self.encoder(x)
        quantized, indices = self.quantizer(e)
        o = self.decoder(quantized)
        
        # rec_loss = F.mse_loss(o, x)
        total_loss = self.custom_l1_loss(o, x)
    
        return total_loss, o
    
    def custom_mse_loss(self, pred, gt):
        real_data_ids = ~torch.all((gt==0), dim=2)

        loss_tensor = F.mse_loss(pred, gt, reduction='none')
        loss_real_values = loss_tensor[real_data_ids.nonzero(as_tuple=True)]
        loss = torch.mean(loss_real_values)

        return loss
    def custom_l1_loss(self, pred, gt):
        real_data_ids = ~torch.all((gt==0), dim=2)

        loss_tensor = F.l1_loss(pred, gt, reduction='none')
        loss_real_values = loss_tensor[real_data_ids.nonzero(as_tuple=True)]
        loss = torch.mean(loss_real_values)

        return loss


    def get_quantize_vectors(self, x):

        e = self.encoder(x)
        quantized, indices = self.quantizer(e)

        return indices, quantized
    
    
    def calculate_perp(self, indices):
        encodings = F.one_hot(indices.to(torch.int64), self.codebook_size).float().reshape(-1, self.codebook_size)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        # cluster_use = torch.sum(avg_probs > 0)
        return perplexity
    
    
    
