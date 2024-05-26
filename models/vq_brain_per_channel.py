import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from vector_quantize_pytorch import ResidualVQ, VectorQuantize

import numpy as np
from einops import rearrange

from vector_quantize_pytorch import FSQ
from dataclasses import dataclass
from simple_parsing import Serializable

@dataclass
class VAEConfig(Serializable):
    C: int = 128
    n_features: int = 1
    levels: tuple = (8, 5, 5, 5)
    stride_list: tuple = (2, 2, 2)


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
            nn.ELU(),
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
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels,
                         dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, 
                         dilation=1),
            nn.ELU(),
            CausalConv1d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=2*stride, 
                         stride=stride),
            nn.ELU()
        )

    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self, C, D, n_electrodes, stride_list):
        super().__init__()
        self.first_layer = CausalConv1d(in_channels=n_electrodes, out_channels=C, kernel_size=3)
        self.act = nn.ELU()
        self.blocks = nn.Sequential(*[EncoderBlock(in_channels=C, out_channels=C, stride=stride) for stride in stride_list])
        self.last_layer = CausalConv1d(in_channels=C, out_channels=D, kernel_size=3)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.act(x)
        x = self.blocks(x)
        x = self.last_layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConvTranspose1d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=2*stride, stride=stride),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, 
                         out_channels=out_channels,
                         dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, 
                         out_channels=out_channels,
                         dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, 
                         out_channels=out_channels,
                         dilation=1),
            nn.ELU()
        )

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, C, D, n_channels_out, stride_list):
        super().__init__()
        self.first_layer = CausalConv1d(in_channels=D, out_channels=C, kernel_size=3)
        self.act = nn.ELU()
        self.blocks = nn.Sequential(*[DecoderBlock(in_channels=C, out_channels=C, stride=stride) for stride in stride_list])
        self.last_layer = CausalConv1d(in_channels=C, out_channels=n_channels_out, kernel_size=3)
    
    def forward(self, x):
        x = self.first_layer(x)
        x = self.act(x)
        x = self.blocks(x)
        x = self.last_layer(x)
        return x


class SoundStream(nn.Module):
    def __init__(self, C, n_features, levels=[8,5,5,5], stride_list=[2, 2]):
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
        self.C = C
        self.D = len(levels)
        self.n_features = n_features
        self.downsample = int(np.prod(stride_list))
        self.encoder = Encoder(C=C, D=self.D, n_electrodes=n_features, stride_list=stride_list)
        self.decoder = Decoder(C=C, D=self.D, n_channels_out=n_features, stride_list=stride_list[::-1])
        
        self.quantizer = FSQ(levels=levels, channel_first=True)

        self.codebook_size = self.quantizer.codebook_size 
        print("self.codebook_size", self.codebook_size)
        print("self.downsample", self.downsample)
   
    def forward(self, x, targets=None, date_info=None,  return_preds=False):
        """
        Params:
            x: is tensor with shape (Batch, Time, Channels)
        Returns:
            total_loss: mse on nonpadded data
            o: is tensor with shape (batch, Time, Channels)
        """
        b=x.size(0)
        x = rearrange(x, 'b t (f c) -> (b c) f t', b=b, f=self.n_features) #4, 768, 256 -> 4x256, 768, 1

        e = self.encoder(x)
        quantized, indices = self.quantizer(e)
        o = self.decoder(quantized)

        total_loss = F.l1_loss(o, x)
        losses = {'total_loss': total_loss}
        # total_loss = self.custom_l1_loss(o, x)
        if return_preds:
            o = rearrange(o, '(b c) f t -> b t (f c)', b=b, f=self.n_features) #4, 768, 256 -> 4x256, 768, 1
            return losses, o
        else:
            return losses, None

    # def forward_new(self, x, targets=None, date_info=None,  return_preds=False):
    #     """
    #     Params:
    #         x: is tensor with shape (Batch, Time, Channels)
    #     Returns:
    #         total_loss: mse on nonpadded data
    #         o: is tensor with shape (batch, Time, Channels)
    #     """
    #     b=x.size(0)
    #     # x = rearrange(x, 'b t (f c) -> (b c) f t', b=b, f=self.n_features) #4, 768, 256 -> 4x256, 768, 1
    #     x = rearrange(x, 'b t (f c) -> c b f t', b=b, f=self.n_features) #4, 768, 256 -> 4x256, 768, 1

    #     outputs = []
    #     for sample in x:
            
    #         e = self.encoder(sample)
    #         # reshaping for quantization
    #         quantized, indices = self.quantizer(e)
    #         o = self.decoder(quantized)
    #         outputs.append(o)

    #     outputs = torch.stack(outputs, dim=0)

    #     total_loss = F.l1_loss(outputs, x)
    #     losses = {'total_loss': total_loss}
    #     # total_loss = self.custom_l1_loss(o, x)
    #     if return_preds:
    #         o = rearrange(o, 'c b f t -> b t (f c)', b=b, f=self.n_features) #4, 768, 256 -> 4x256, 768, 1
    #         return losses, o
    #     else:
    #         return losses, None
    def get_indices(self, x, return_embeddings=False):
        b = x.size(0)
        x = rearrange(x, 'b t (f c) -> (b c) f t', b=b, f=self.n_features)
        
        # print('x.shape: (b c) f t', x.shape)
        
        e = self.encoder(x)
        embeds, indices = self.quantizer(e)

        # print('embeds', embeds.shape)

        embeds = rearrange(embeds, '(b c) d t -> b t c d', b=b).contiguous()
        indices = rearrange(indices, '(b c) t -> b t c', b=b).contiguous()
        
        # print('after arrange embeds.', embeds.shape)
        
        if return_embeddings:
            return indices, embeds 
        return indices
        
    def decode_indices(self, indices):
        """
        indices: (b t c)
        return: (b t (c f))
        """
        b = indices.size(0)
        
        indices  = rearrange(indices, 'b t c -> (b c) t')
        quantized = self.quantizer.indices_to_codes(indices)    
        reconstruct = self.decoder(quantized)
        reconstruct = rearrange(reconstruct, '(b c) f t -> b t (f c)', b=b, c=256, f=self.n_features)
        return reconstruct
        
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
    
    def calculate_perp(self, indices):
        encodings = F.one_hot(indices.to(torch.int64), self.codebook_size).float().reshape(-1, self.codebook_size)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        # cluster_use = torch.sum(avg_probs > 0)
        return perplexity
    
    
    
