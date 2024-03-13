"""
Contains some code from:
https://github.com/Sentdex/GANTheftAuto
with the following license:

Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/GameGAN_code.
Authors: Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, Sanja Fidler
"""


from torch import nn
import torch
from torch.nn import functional as F
from torch.nn import init
import utils
#import sys
#sys.path.append('..')

try:
    from models import modules
except:
    import modules
import functools


# same encoder architecture as GameGAN
def choose_netG_encoder(args):
    '''
    image input encoder
    '''
    if args.img_size[0] == 64:
        last_dim = args.enc_base_channel
        enc = nn.Sequential(
            nn.Conv2d(args.in_channel, args.enc_base_channel // 8, 4, 1, 1), #[2, 3, 64, 64]->[2, 64, 63, 63]
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 8, args.enc_base_channel // 8, 3, 2), #[2, 64, 63, 63]->[2, 64, 31, 31]
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 8, args.enc_base_channel // 8, 3, 2), #[2, 64, 31, 31]->[2, 64, 15, 15] 
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 8, args.enc_base_channel // 8, 3, 2), #[2, 64, 15, 15]->[2, 64, 7, 7]
            nn.LeakyReLU(0.2),
            modules.View((-1, (args.enc_base_channel // 8) * 7 * 7)), #[2, 64, 7, 7]->[2, 3136]
            nn.Linear((args.enc_base_channel // 8) * 7 * 7, last_dim), #[2, 3136]->[2, 512]
            nn.LeakyReLU(0.2),
        )
    elif args.img_size[0] == 88 and args.img_size[1] == 64:
        last_dim = args.enc_base_channel
        enc = nn.Sequential(
            nn.Conv2d(args.in_channel, args.enc_base_channel // 8, 4, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 8, args.enc_base_channel // 8, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 8, args.enc_base_channel // 4, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 4, args.enc_base_channel // 4, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 4, args.enc_base_channel // 2, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 2, args.enc_base_channel // 8, 3, 2, 1),
            nn.LeakyReLU(0.2),
            modules.View((-1, (args.enc_base_channel // 8) * 11 * 8)),
            nn.Linear((args.enc_base_channel // 8) * 11 * 8, last_dim),
            nn.LeakyReLU(0.2),
        )
    elif args.img_size[0] == 48 and args.img_size[1] == 80:
        last_dim = args.enc_base_channel
        enc = nn.Sequential(
            nn.Conv2d(args.in_channel, args.enc_base_channel // 8, 4, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 8, args.enc_base_channel // 8, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 8, args.enc_base_channel // 4, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 4, args.enc_base_channel // 4, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 4, args.enc_base_channel // 2, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 2, args.enc_base_channel // 8, 3, 2, 1),
            nn.LeakyReLU(0.2),
            modules.View((-1, (args.enc_base_channel // 8) * 6 * 10)),
            nn.Linear((args.enc_base_channel // 8) * 6 * 10, last_dim),
            nn.LeakyReLU(0.2),
        )
    elif args.img_size[0] == 256:
        last_dim = args.enc_base_channel
        enc = nn.Sequential(
            nn.Conv2d(args.in_channel, args.enc_base_channel // 16, 4, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 16, args.enc_base_channel // 8, 3, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 8, args.enc_base_channel // 4, 3, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 4, args.enc_base_channel // 2, 3, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel // 2, args.enc_base_channel, 3, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(args.enc_base_channel, args.enc_base_channel // 8, 3, 2),
            nn.LeakyReLU(0.2),
            modules.View((-1, (args.enc_base_channel // 8) * 7 * 7)),
            nn.Linear((args.enc_base_channel // 8) * 7 * 7, last_dim),
            nn.LeakyReLU(0.2),
        )
    else:
        print(args.img_size)
        assert 0, 'model-%s not supported'

    return enc


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64'):
    arch = {}
    # (width)x(height)
    arch['64x64'] = {'in_channels': [ch * item for item in [16, 8, 4]],
                     'out_channels': [ch * item for item in [8, 4, 2]],
                     'upsample': [True] * 3,
                     'resolution': [16, 32, 64],
                     'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                   for i in range(4, 7)}}
    arch['64x88'] = {'in_channels' :  [ch * item for item in [16, 8, 4, 2, 2]],
                     'out_channels' : [ch * item for item in [8, 4, 2, 2, 2]],
                     'upsample' : [2] * 3 + [1] * 2,
                     'resolution' : [16, 32, 64, 128, 256],
                     'attention' : {res: (res in [int(item) for item in attention.split('_')])
                                    for res in [16, 32, 64, 128, 256]}}
    arch['80x48'] = {'in_channels' :  [ch * item for item in [16, 8, 4, 2, 2]],
                     'out_channels' : [ch * item for item in [8, 4, 2, 2, 2]],
                     'upsample' : [2] * 3 + [1] * 2,
                     'resolution' : [16, 32, 64, 128, 256],
                     'attention' : {res: (res in [int(item) for item in attention.split('_')])
                                    for res in [16, 32, 64, 128, 256]}}
    arch['256x256'] = {'in_channels' :  [ch * item for item in [16, 8, 4, 2, 2]],
                       'out_channels' : [ch * item for item in [8, 4, 2, 2, 1]],
                       'upsample' : [True] * 6,
                       'resolution' : [16, 32, 64, 128, 256],
                       'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                      for i in range(4, 9)}}
    return arch


class Decoder(nn.Module):
    def __init__(self, G_ch=64, bottom_width=(8, 8), resolution=(64, 64),
                 G_kernel_size=3, G_attn='64', num_G_SVs=1, num_G_SV_itrs=1,
                 hier=False, G_activation=nn.ReLU(inplace=False),
                 SN_eps=1e-12, G_init='ortho', skip_init=False,
                 G_param='SN', args=None, summary=False):
        super(Decoder, self).__init__()
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.dim_z = args.dec_hidden_dim
        # The initial spatial dimensions and Channel width mulitplier
        if (resolution[0] == 88 and resolution[1] == 64):
            bottom_width = (11, 8)
        elif (resolution[0] == 48 and resolution[1] == 80):
            bottom_width = (6, 10)
            self.ch = 48
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = '64_32' if (resolution[0] == 88 and resolution[1] == 64) \
                         or (resolution[0] == 48 and resolution[1] == 80) else G_attn
        # Hierarchical latent space?
        self.hier = hier
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[f'{resolution[1]}x{resolution[0]}']
        self.args = args
        self.summary = summary

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.dim_z // self.num_slots)
            # Recalculate latent dimensionality for even splitting into chunks
            self.dim_z = self.z_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':

            self.which_conv = functools.partial(modules.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(modules.SNLinear,
                                                  num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                  eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # Prepare model
        in_dim = self.dim_z
        self.linear = self.which_linear(in_dim,
                            self.arch['in_channels'][0] * (self.bottom_width[0] * self.bottom_width[1]))
        
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            upsample_factor = 2 if type(self.arch['upsample'][index]) is bool else self.arch['upsample'][index]
            if index == 0:
                self.in_dim = self.arch['in_channels'][index]
                in_dim = self.in_dim
                if resolution[0] == 84:
                    upsample_factor = 3
            else:
                in_dim = self.arch['in_channels'][index]

            self.blocks += [[modules.GBlock(in_channels=in_dim,
                                            out_channels=self.arch['out_channels'][index],
                                            which_conv=self.which_conv,
                                            activation=self.activation,
                                            upsample=(functools.partial(F.interpolate, scale_factor=upsample_factor)
                                                        if self.arch['upsample'][index] else None))]]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]] and \
                                (not utils.check_arg(self.args, 'no_attention')):
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [modules.Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.all_blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        last_conv = self.which_conv

        if self.args.no_in:
            self.output_layer = nn.Sequential(self.activation,
                                                    last_conv(self.arch['out_channels'][-1], 3))
        else:
            self.output_layer = nn.Sequential(nn.InstanceNorm2d(self.arch['out_channels'][-1]),
                                                self.activation,
                                                last_conv(self.arch['out_channels'][-1], 3))

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    def forward(self, z):
        # simple rendering engine
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width[0], self.bottom_width[1])
        for blocklist in self.all_blocks:
            for block in blocklist:
                h = block(h)
        return torch.tanh(self.output_layer(h))


class EncoderDecoder(nn.Module):
    def __init__(self, args):
        super(EncoderDecoder, self).__init__()
        self.enc = choose_netG_encoder(args)
        self.dec = Decoder(G_ch=args.nfilterDec, args=args, resolution=args.img_size)

    def forward(self, img):
        return self.dec(self.enc(img))


if __name__ == '__main__':
    import torch
    from torchinfo import summary
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    class Args:
        def __init__(self, img_size, nfilterDec=32, in_channel=3, enc_base_channel=512,
                     dec_hidden_dim=512, no_in=True):
            self.img_size = img_size
            self.in_channel = in_channel
            self.enc_base_channel = enc_base_channel
            self.dec_hidden_dim = dec_hidden_dim
            self.hidden_dim = dec_hidden_dim
            self.no_in = no_in
            self.nfilterDec = nfilterDec

    
    img_sizes=[(64, 64), (48, 80)]
    nfilterDecs = [32, 48]
    #nfilterDecs = [32, 32]
    for img_size, nfilterDec in zip(img_sizes, nfilterDecs):
        print('\n\n----------------------------------------------------------------------------')
        args = Args(img_size, nfilterDec)
        x_size = (2, 3, *img_size)
        z_size = (2, args.dec_hidden_dim)
        x = torch.randn(*x_size)
        z = torch.randn(*z_size)
        img_enc = choose_netG_encoder(args)
        summary(img_enc, input_size=x_size, col_names=["input_size", "output_size", "num_params"])
        print('\n\n----------------------------------------------------------------------------')
        encoded_img = img_enc(x)
        print('output shape of img enc:', encoded_img.shape)
        print('\n\n----------------------------------------------------------------------------')
        img_dec = Decoder(G_ch=args.nfilterDec, args=args, resolution=args.img_size)
        summary(img_dec, input_size=z_size, col_names=["input_size", "output_size", "num_params"])
        print('\n\n----------------------------------------------------------------------------')
        print('output shape of img dec:', img_dec(encoded_img).shape)
        encdec = EncoderDecoder(args)
        print('output shape of img encdec:', encdec(x).shape)