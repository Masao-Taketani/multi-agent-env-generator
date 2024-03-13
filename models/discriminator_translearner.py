"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

import torch
from torch import nn
import torch.utils.data
import torch.utils.data.distributed
import utils
import sys
sys.path.append('..')

try:
    from models.modules import View
    from models import model_utils
    from models import layers
except:
    from modules import View
    import model_utils
    import layers

import functools


class DiscriminatorSingleLatent(nn.Module):
    def __init__(self, opts):
        super(DiscriminatorSingleLatent, self).__init__()
        self.opts = opts
        dim = opts.nfilterD * 16

        self.which_linear = functools.partial(layers.SNLinear,
                                              num_svs=1, num_itrs=1,
                                              eps=1e-12)

        sinput_dim = opts.latent_z_size
        l = [self.which_linear(sinput_dim, dim)]
        l.append(nn.BatchNorm1d(dim)),
        l.append(nn.LeakyReLU(0.2))

        num_layers = 3

        for _ in range(num_layers):
            l.append(self.which_linear(dim, dim))
            l.append(nn.BatchNorm1d(dim))
            l.append(nn.LeakyReLU(0.2))
        self.base = nn.Sequential(*l)

        self.d_final = nn.Sequential(self.which_linear(dim, dim),
                                     nn.BatchNorm1d(dim),
                                     nn.LeakyReLU(0.2),
                                     self.which_linear(dim, 1))


    def forward(self, x):

        h = self.base(x)
        return self.d_final(h), h


class Discriminator(nn.Module):

    def __init__(self, opts):
        super(Discriminator, self).__init__()

        self.opts = opts
        f_size = 4
        orig_mul_for_a2f = 16
        assert orig_mul_for_a2f % opts.num_agents == 0, f'num_agents has to be a divisor of {orig_mul_for_a2f}'
        adjusted_mul_for_a2f = orig_mul_for_a2f // opts.num_agents

        self.ds = DiscriminatorSingleLatent(opts)
        conv3d_dim = opts.nfilterD_temp * 16

        self.temporal_window = self.opts.config_temporal

        self.conv3d, self.conv3d_final = \
            model_utils.choose_netD_temporal_m(
                self.opts, conv3d_dim, window=self.temporal_window
            )
        self.conv3d = nn.ModuleList(self.conv3d)
        self.conv3d_final = nn.ModuleList(self.conv3d_final)

        self.which_conv = functools.partial(layers.SNConv2d,
                                            kernel_size=f_size, padding=0,
                                            num_svs=1, num_itrs=1,
                                            eps=1e-12)
        self.which_linear = functools.partial(layers.SNLinear,
                                              num_svs=1, num_itrs=1,
                                              eps=1e-12)

        # For action discriminator
        self.trans_conv = self.which_linear(opts.nfilterD*16*2, opts.nfilterD*16)
        self.to_transition_feature = nn.Sequential(self.trans_conv,
                                                   nn.LeakyReLU(0.2),
                                                   View((-1, opts.nfilterD*16)))

        action_space = self.opts.action_space
        self.action_to_feats = nn.ModuleList([nn.Linear(action_space, opts.nfilterD*adjusted_mul_for_a2f) \
                                              for _ in range(opts.num_agents)])

        self.reconstruct_actions = nn.ModuleList([self.which_linear(opts.nfilterD*16, action_space) \
                                              for _ in range(opts.num_agents)])

    def forward(self, images, actions, states, warm_up, neg_actions=None):
        """
        images: (seq, bs, feat)
        actions: (seq, bs, num_classes)
        states: (seq+1, bs, feat)
        """
        dout = {}
        neg_content_predictions = None
        seq_len, batch_size, _ = actions[0].shape

        if warm_up is None or warm_up == 0:
            warm_up = 1 # even if warm_up is 0, the first screen is from GT            
        #gt_states = torch.cat(states[:warm_up], dim=0)
        gt_states = states.reshape(batch_size * (seq_len + 1), -1)[:warm_up * batch_size]
        images = images.reshape(batch_size * seq_len, -1)
        actions = [act.reshape(batch_size * seq_len, -1) for act in actions]
        if neg_actions is not None:
            neg_actions = [neg_act.reshape(batch_size * seq_len, -1) for neg_act in neg_actions]
        #print('gt_states:', gt_states.device)
        #print('images:', images.device)
        #print('actions[0]:', actions[0].shape)
        
        # single_frame_predictions_all may not be needed since current decoder can reconstruct 
        # original images pefectly (may need to delete after experiment)
        single_frame_predictions_all, tmp_features = self.ds(torch.cat([gt_states, images], dim=0))
        single_frame_predictions_all = single_frame_predictions_all[warm_up * batch_size:]
        frame_features = tmp_features[warm_up*batch_size:]
        next_features = frame_features
        # action discriminator
        prev_frames = torch.cat([tmp_features[:warm_up*batch_size],
                                tmp_features[(warm_up+warm_up-1)*batch_size:-batch_size]], dim=0)

        if self.opts.input_detach:
            prev_frames = prev_frames.detach()
        # shape of transition_features: (bs * seq, dim * 2)
        transition_features = self.to_transition_feature(torch.cat([prev_frames, next_features], dim=1))
        #action_features = self.action_to_feat(torch.cat(actions[:-1], dim=0))
        action_features = [self.action_to_feats[i](act) for i, act in enumerate(actions)]
        if neg_actions is not None:
            #neg_action_features = self.action_to_feat(torch.cat(neg_actions[:-1], dim=0))
            neg_action_features = [self.action_to_feats[i](neg_act) for i, neg_act in enumerate(neg_actions)]

        action_recons = [self.reconstruct_actions[i](transition_features) for i in range(len(self.reconstruct_actions))]

        temporal_predictions = []
        stacked = torch.cat([*action_features, transition_features], dim=1)
        # reshape from (bs*seq_len, feat) to (seq_len, bs, feat) 
        # and permute from (seq_len, bs, feat) to (bs, seq_len, feat)
        stacked = stacked.view(seq_len, batch_size, -1).permute(1,0,2)
        # permute from (bs, seq_len, feat) to (bs, feat, seq_len) for conv2d
        stacked = stacked.permute(0, 2, 1)

        if neg_actions is not None:
            neg_stacked = torch.cat([*neg_action_features, transition_features], dim=1)
            neg_stacked = neg_stacked.view(seq_len, batch_size, -1).permute(1, 0, 2)
            neg_stacked = neg_stacked.permute(0, 2, 1)
            if self.opts.do_latent:
                # unsqueeze from (bs, feat, seq_len) to (bs, feat, seq_len, 1) for conv2d
                neg_stacked = neg_stacked.unsqueeze(-1)

            neg_content_predictions = []
            aa = self.conv3d[0](neg_stacked)
            a_out = self.conv3d_final[0](aa)
            neg_content_predictions.append(a_out.view(batch_size, -1))
            if self.temporal_window >= 12:
                bb = self.conv3d[1](aa)
                b_out = self.conv3d_final[1](bb)
                neg_content_predictions.append(b_out.view(batch_size, -1))
            if self.temporal_window >= 18:
                cc = self.conv3d[2](bb)
                c_out = self.conv3d_final[2](cc)
                neg_content_predictions.append(c_out.view(batch_size, -1))
            if self.temporal_window >= 30:
                dd = self.conv3d[3](cc)
                d_out = self.conv3d_final[3](dd)
                neg_content_predictions.append(d_out.view(batch_size, -1))

        # unsqueeze from (bs, feat, seq_len) to (bs, feat, seq_len, 1) for conv2d
        stacked = stacked.unsqueeze(-1)

        aa = self.conv3d[0](stacked)
        a_out = self.conv3d_final[0](aa)
        temporal_predictions.append(a_out.view(batch_size, -1))
        if self.temporal_window >= 12:
            bb = self.conv3d[1](aa)
            b_out = self.conv3d_final[1](bb)
            temporal_predictions.append(b_out.view(batch_size, -1))
        if self.temporal_window >= 18:
            cc = self.conv3d[2](bb)
            c_out = self.conv3d_final[2](cc)
            temporal_predictions.append(c_out.view(batch_size, -1))
        if self.temporal_window >= 36:
            dd = self.conv3d[3](cc)
            d_out = self.conv3d_final[3](dd)
            temporal_predictions.append(d_out.view(batch_size, -1))

        #dout['disc_features'] = frame_features[:(len(states)-1)*batch_size]
        dout['disc_features'] = frame_features
        dout['single_frame_predictions_all'] = single_frame_predictions_all
        dout['content_predictions'] = temporal_predictions
        dout['neg_content_predictions'] = neg_content_predictions
        dout['action_recons'] = action_recons
        return dout

    def update_opts(self, opts):
        self.opts = opts
        return


if __name__ == '__main__':
    import torch
    from torchinfo import summary
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    class Args:
        def __init__(self, max_seq_len=64, num_agents=2):
            self.nfilterD = 64
            self.latent_z_size = 512
            self.nfilterD_temp = max_seq_len
            self.config_temporal = max_seq_len
            self.action_space = 4
            self.input_detach = True
            self.do_latent = True
            self.bs = 8
            self.max_seq_len = max_seq_len
            self.num_agents = num_agents

    
    class DiscTempForSummary(nn.Module):

        def __init__(self, opts):
            super(DiscTempForSummary, self).__init__()

            self.opts = opts
            f_size = 4
            orig_mul_for_a2f = 16
            assert orig_mul_for_a2f % opts.num_agents == 0, f'num_agents has to be a divisor of {orig_mul_for_a2f}'
            adjusted_mul_for_a2f = orig_mul_for_a2f // opts.num_agents

            conv3d_dim = opts.nfilterD_temp * 16

            self.temporal_window = self.opts.config_temporal

            self.conv3d, self.conv3d_final = \
                model_utils.choose_netD_temporal_m(
                    self.opts, conv3d_dim, window=self.temporal_window
                )
            self.conv3d = nn.ModuleList(self.conv3d)
            self.conv3d_final = nn.ModuleList(self.conv3d_final)

            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=f_size, padding=0,
                                                num_svs=1, num_itrs=1,
                                                eps=1e-12)
            self.which_linear = functools.partial(layers.SNLinear,
                                                num_svs=1, num_itrs=1,
                                                eps=1e-12)

            # For action discriminator
            self.trans_conv = self.which_linear(opts.nfilterD*16*2, opts.nfilterD*16)
            self.to_transition_feature = nn.Sequential(self.trans_conv,
                                                    nn.LeakyReLU(0.2),
                                                    View((-1, opts.nfilterD*16)))

            action_space = self.opts.action_space
            self.action_to_feats = nn.ModuleList([nn.Linear(action_space, opts.nfilterD*adjusted_mul_for_a2f) \
                                                for _ in range(opts.num_agents)])

            self.reconstruct_actions = nn.ModuleList([self.which_linear(opts.nfilterD*16, action_space) \
                                                for _ in range(opts.num_agents)])

        def forward(self, tmp_features, stacked, neg_stacked=None):
            """
            tmp_features: (bs * (warm_up + max_seq_len), latent_z_size * 2)
            stacked: (bs * max_seq_len, (orig_mul_for_a2f * nfilterD) * 2)
            neg_stacked: (bs * max_seq_len, (orig_mul_for_a2f * nfilterD) * 2)
            """
            batch_size, seq_len, warm_up = self.opts.bs, self.opts.max_seq_len, 1
            neg_content_predictions = None
            frame_features = tmp_features[warm_up*batch_size:]
            next_features = frame_features
            # action discriminator
            prev_frames = torch.cat([tmp_features[:warm_up*batch_size],
                                    tmp_features[(warm_up+warm_up-1)*batch_size:-batch_size]], dim=0)

            if self.opts.input_detach:
                prev_frames = prev_frames.detach()

            temporal_predictions = []
            # reshape from (bs*seq_len, feat) to (seq_len, bs, feat) 
            # and permute from (seq_len, bs, feat) to (bs, seq_len, feat)
            stacked = stacked.view(seq_len, batch_size, -1).permute(1,0,2)
            # permute from (bs, seq_len, feat) to (bs, feat, seq_len) for conv2d
            stacked = stacked.permute(0, 2, 1)

            if neg_stacked is not None:
                # Conditioned on negative actions
                neg_stacked = neg_stacked.view(seq_len, batch_size, -1).permute(1, 0, 2)
                neg_stacked = neg_stacked.permute(0, 2, 1)
                if self.opts.do_latent:
                    # unsqueeze from (bs, feat, seq_len) to (bs, feat, seq_len, 1) for conv2d
                    neg_stacked = neg_stacked.unsqueeze(-1)

                neg_content_predictions = []
                aa = self.conv3d[0](neg_stacked)
                a_out = self.conv3d_final[0](aa)
                neg_content_predictions.append(a_out.view(batch_size, -1))
                if self.temporal_window >= 12:
                    bb = self.conv3d[1](aa)
                    b_out = self.conv3d_final[1](bb)
                    neg_content_predictions.append(b_out.view(batch_size, -1))
                if self.temporal_window >= 18:
                    cc = self.conv3d[2](bb)
                    c_out = self.conv3d_final[2](cc)
                    neg_content_predictions.append(c_out.view(batch_size, -1))
                if self.temporal_window >= 30:
                    dd = self.conv3d[3](cc)
                    d_out = self.conv3d_final[3](dd)
                    neg_content_predictions.append(d_out.view(batch_size, -1))

            # Conditioned on positive actions
            # unsqueeze from (bs, feat, seq_len) to (bs, feat, seq_len, 1) for conv2d
            stacked = stacked.unsqueeze(-1)

            aa = self.conv3d[0](stacked)
            a_out = self.conv3d_final[0](aa)
            temporal_predictions.append(a_out.view(batch_size, -1))
            if self.temporal_window >= 12:
                bb = self.conv3d[1](aa)
                b_out = self.conv3d_final[1](bb)
                temporal_predictions.append(b_out.view(batch_size, -1))
            if self.temporal_window >= 18:
                cc = self.conv3d[2](bb)
                c_out = self.conv3d_final[2](cc)
                temporal_predictions.append(c_out.view(batch_size, -1))
            if self.temporal_window >= 36:
                dd = self.conv3d[3](cc)
                d_out = self.conv3d_final[3](dd)
                temporal_predictions.append(d_out.view(batch_size, -1))

            if neg_stacked is not None:
                return temporal_predictions, neg_content_predictions
            else:
                return temporal_predictions
    

    warm_up = None
    orig_mul_for_a2f = 16
    max_seq_lens = [64, 128]
    num_agents = 2
    for max_seq_len in max_seq_lens:
        print(f'[max_seq_len]: {max_seq_len}')
        print(f'[num_agents]: {num_agents}')
        args = Args(max_seq_len, num_agents)
        disc = Discriminator(args)
        # inputs for DriveGAN format
        #images = torch.randn(bs * (time_steps - 1), 512)
        #actions = []
        #states = []
        #neg_actions = []
        #for _ in range(time_steps):
        #    actions.append(torch.randn(bs, action_space))
        #    states.append(torch.randn(bs, dim))
        #    neg_actions.append(torch.randn(bs, action_space))
        
        print('\n\n----------------------------------------------------------------------------')
        ds = DiscriminatorSingleLatent(args)
        ds_inp_size = (args.max_seq_len * args.bs, args.latent_z_size)
        summary(ds, input_size=ds_inp_size, col_names=["input_size", "output_size", "num_params"])

        dt = DiscTempForSummary(args)
        """
        tmp_features: (bs * (warm_up + max_seq_len), latent_z_size * 2)
        stacked: (bs * max_seq_len, (orig_mul_for_a2f * nfilterD) * 2)
        neg_stacked: (bs * max_seq_len, (orig_mul_for_a2f * nfilterD) * 2)
        """
        ds_inp_size = [(args.bs * (1 + args.max_seq_len), args.latent_z_size * 2),
                       (args.bs * args.max_seq_len, (orig_mul_for_a2f * args.nfilterD) * 2),]
                       #(args.bs * args.max_seq_len, (orig_mul_for_a2f * args.nfilterD) * 2),]
        summary(dt, input_size=ds_inp_size, col_names=["input_size", "output_size", "num_params"])
        print('\n\n----------------------------------------------------------------------------')

        images = torch.randn(args.max_seq_len, args.bs, args.latent_z_size)
        actions = [torch.randn(args.max_seq_len, args.bs, args.action_space) \
                                    for _ in range(args.num_agents)]
        neg_actions = [torch.randn(args.max_seq_len, args.bs, args.action_space) \
                                    for _ in range(args.num_agents)]
        states = torch.randn(args.max_seq_len+1, args.bs, args.latent_z_size)
        results = disc(images, actions, states, warm_up, neg_actions=neg_actions)
        for k, v in results.items():
            if torch.is_tensor(v):
                print(f'{k}:', v.shape)
            else:
                print(f'{k}:')
                for elem in v:
                    print(elem.shape)