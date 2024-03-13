"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

"""
Contains some code from:
https://github.com/LMescheder/GAN_stability
with the following license:
MIT License

Copyright (c) 2018 Lars Mescheder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import utils
import torch
import torch.nn.functional as F
import torch.utils.data
import math


class Trainer(object):
    def __init__(self, opts, netG, netD, optG, optD, reg_param):

        self.opts = opts

        self.netG = netG
        self.netG.opts = opts
        self.netD = netD
        if self.netD is not None:
            self.netD.opts = opts

        self.optG = optG
        self.optD = optD

        self.reg_param = reg_param

        ## Default to hinge loss
        #if utils.check_arg(opts, 'standard_gan_loss'):
        #    self.generator_loss = self.standard_gan_loss
        #    self.discriminator_loss = self.standard_gan_loss
        #else:
        #    self.generator_loss = self.loss_hinge_gen
        #    self.discriminator_loss = self.loss_hinge_dis

        self.generator_loss = self.loss_hinge_gen
        self.discriminator_loss = self.loss_hinge_dis

        if self.opts.feat_loss == 'l1':
            self.feat_loss_criterion = F.l1_loss
        elif self.opts.feat_loss == 'l2':
            self.feat_loss_criterion = F.mse_loss
        else:
            raise KeyError(f'feat_loss has to be either l1 or l2. \
                             What we get: {self.opts.feat_loss}')
        
        if self.opts.recon_loss == 'l1':
            self.recon_loss_criterion = F.l1_loss
        elif self.opts.recon_loss == 'l2':
            self.recon_loss_criterion = F.mse_loss
        else:
            raise KeyError(f'recon_type has to be either l1 or l2. \
                             What we get: {self.opts.recon_loss}')

        if self.opts.continuous_action:
            self.act_recon_criterion = self.recon_loss_criterion
        else:
            self.act_recon_criterion = F.cross_entropy

        self.adjust_action_for_disc_func = self.adjust_continuous_actions_for_disc if self.opts.continuous_action \
                                           else self.adjust_discrete_actions_for_disc

    # Hinge loss for discriminator
    def loss_hinge_dis(self, logits, label, masking=None, div=None):
        if label == 1:
            t = F.relu(1. - logits)
        else:
            t = F.relu(1. + logits)

        if div is None:
            return torch.mean(t)
        else:
            assert(masking is not None)
            t = t * masking
            return torch.sum(t) / div

    # Hinge loss for generator
    def loss_hinge_gen(self, dis_fake):
        loss = -torch.mean(dis_fake)
        return loss

    # BCE GAN loss
    def standard_gan_loss(self, d_out, target=1):
        if d_out is None:
            return utils.check_gpu(self.opts.gpu, torch.FloatTensor([0]))
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        loss = F.binary_cross_entropy_with_logits(d_out, targets)
        return loss

    # Reconstruction loss
    def get_recon_loss(self, input, target, detach=True, div=None):
        """
        shape of input & target: (max_seq_len, bs, dim)
        """
        if div is None:
            div = target.size(0)
        if detach:
            target = target.detach()
        # intentionally leave div as max_seq_len as the original code presented
        if self.opts.seq_weights == True:
            # multiply 2.0 to get the same expected loss as the original loss (may erase it later)
            seq_weights = torch.tensor([2.0*(i+1)/self.opts.max_seq_len for i in range(self.opts.max_seq_len)]
                                      ).reshape(self.opts.max_seq_len, 1, 1).to(self.opts.device)
            loss = self.recon_loss_criterion(input, target, reduction='none') * seq_weights
            loss = loss.sum() / div
        else:
            loss = self.recon_loss_criterion(input, target, reduction='sum') / div
        ## for loss check
        ##loss = self.recon_loss_criterion(input, target, reduction='none')
        return loss

    def merge_dicts(self, d1, d2, d2name=''):
        for key, val in d2.items():
            d1[d2name+'_'+key] = val
        return d1

    def get_num_repeat(self, input_like, bs):
        return input_like.size(0) // bs

    def calculate_discriminator_adv_loss_fake(self, loss_dict, dout_fake):
        ## temporal loss
        dloss_fake_content_loss = 0
        if self.opts.temporal_hierarchy:
            for i in range(len(dout_fake['content_predictions'])):
                curloss = self.discriminator_loss(dout_fake['content_predictions'][i], 0)
                loss_dict['dloss_fake_content_loss' + str(i)] = curloss
                dloss_fake_content_loss += curloss
            dloss_fake_content_loss = dloss_fake_content_loss / len(dout_fake['content_predictions'])
            loss_dict['dloss_fake_content_loss'] = dloss_fake_content_loss

        ## single frame loss
        dloss_fake_single_frame_loss = self.discriminator_loss(dout_fake['single_frame_predictions_all'], 0)
        loss_dict['dloss_fake_single_frame_loss'] = dloss_fake_single_frame_loss

        loss = dloss_fake_content_loss + dloss_fake_single_frame_loss


        return loss_dict, loss
    
    def calculate_action_recon_loss(self, dout, actions, batch_size):
        acts_reshape = [act.reshape(self.opts.max_seq_len*batch_size, -1) for act in actions]
        if self.opts.continuous_action:
            self.act_recon_criterion = F.mse_loss
            acts_label = acts_reshape
        else:
            self.act_recon_criterion = F.cross_entropy
            acts_label = [torch.max(act, 1)[1] for act in acts_reshape]
        
        actions_recon_loss = [self.act_recon_criterion(pred, label) for pred, label in zip(dout['action_recons'], acts_label)]
        return actions_recon_loss
        

    def calculate_generator_adv_loss(self, loss_dict, dout_fake, actions, batch_size):
        ## temporal loss
        gloss_content_loss = 0
        if self.opts.temporal_hierarchy:
            for i in range(len(dout_fake['content_predictions'])):
                curloss = self.generator_loss(dout_fake['content_predictions'][i])
                loss_dict['gloss_content_loss' + str(i)] = curloss
                # for loss check
                #if i == 3:
                #    loss_dict['tmp3'] = -dout_fake['content_predictions'][i]
                gloss_content_loss += curloss

        ## single frame loss
        gloss_single_frame_loss = self.generator_loss(dout_fake['single_frame_predictions_all'])
        loss_dict['gloss_single_frame_loss'] = gloss_single_frame_loss

        ## loss to recover action from frames
        actions_recon_loss = self.calculate_action_recon_loss(dout_fake, actions, batch_size)
        
        for i in range(len(actions_recon_loss)):
            loss_dict[f'g_action{i}_recon_loss'] = actions_recon_loss[i]

        total_loss = self.opts.gen_content_loss_multiplier * gloss_content_loss + \
                      gloss_single_frame_loss + sum(actions_recon_loss)

        return loss_dict, total_loss

    def get_kl_loss(self, gout, name, beta, loss_dict):
        kl_loss = gout[name].mean()
        loss_dict[name] = kl_loss
        return kl_loss * beta
    
    def adjust_discrete_actions_for_disc(self, actions, is_bs_first=True):
        # make one-hot and permute actions for the discriminator input
        # (bs, seq) -> (bs, seq, action_space) -> (seq, bs, action_space)
        actions = F.one_hot(actions, num_classes=self.opts.action_space)
        actions = actions.to(torch.float32)
        if is_bs_first: actions = actions.permute(1, 0, 2)
        return actions
    
    def adjust_continuous_actions_for_disc(self, actions, is_bs_first=True):
        # (bs, seq, action_space) -> (seq, bs, action_space)
        if is_bs_first: actions = actions.permute(1, 0, 2)
        return actions

    def adjust_inputs_for_disc(self, real_or_gen_imgs, states, actions, neg_actions=None, is_bs_first=True):
        if is_bs_first:
            # permute gout and states from (bs, seq, feat) to (seq, bs, feat) for the discriminator input
            real_or_gen_imgs = real_or_gen_imgs.permute(1, 0, 2)
            states = states.permute(1, 0, 2)
            
        # Since real images come from states, which contains max_seq_len + 1 steps to include the last target,
        # if the inputs are real images, starting index has to be 1 instead of 0. 
        if real_or_gen_imgs.size(0) == self.opts.max_seq_len+1: real_or_gen_imgs = real_or_gen_imgs[1:]
        actions = [self.adjust_action_for_disc_func(action) for action in actions]
        rtn = (real_or_gen_imgs, states, actions)

        if neg_actions is not None:
            neg_actions = [self.adjust_action_for_disc_func(neg_action) for neg_action in neg_actions]
            rtn = (real_or_gen_imgs, states, actions, neg_actions) 
        
        return rtn
    
    def generator_trainstep(self, states, actions, warm_up, train=True, it=0, mems=None):
        '''
        Run one step of generator

        shape of args:
            if self.opts.is_bs_first == True:
                states: (bs, seq_len + 1, feat)
                actions: (bs, seq_len)
            else:
                states: (seq_len + 1, bs, feat)
                actions: (seq_len, bs)
        '''
        batch_size = states.size(0) if self.opts.is_bs_first else states.size(1)

        if warm_up is not None:
            # set number of warm up images
            if self.opts.warmup_decay_step > 0:
                warm_up = max(self.opts.min_warmup, 
                              math.ceil(warm_up * (1 - it * 1.0 / self.opts.warmup_decay_step)))

        utils.toggle_grad(self.netG, True)
        utils.toggle_grad(self.netD, True)
        if train:
            self.netG.train()
            self.netD.train()
        else:
            self.netG.eval()
            self.netD.eval()

        self.optD.zero_grad()
        self.optG.zero_grad()

        loss_dict = {}
        #gen_actions = actions

        # generate the output sequence
        if warm_up is None:
            gen_inputs = states[:, :-1, :] if self.opts.is_bs_first else states[:-1, :, :]
            if mems is None:        
                gout = self.netG(gen_inputs, actions)
            else:
                gout, mems = self.netG(gen_inputs, actions, mems)
        else:
            raise ValueError("Need to take care of Epoch")
            gout = self.netG(states, actions, warm_up, train=train, epoch=epoch)

        total_loss, dout_fake = 0, None
        ######################### fool discriminator ########################
        # generated sequence
        #gen_adv_input = torch.cat(gout['outputs'], dim=0)
        gen_adv_input, states, actions = self.adjust_inputs_for_disc(gout, states, actions,
                                                                     is_bs_first=self.opts.is_bs_first)

        dout_fake = self.netD(gen_adv_input, actions, states, warm_up)

        # real sequence
        din = states[1:len(gen_adv_input)+1]
        dout_real = self.netD(din, actions, states, warm_up)

        loss_dict, total_loss = self.calculate_generator_adv_loss({}, dout_fake, actions, batch_size)

        if self.opts.disc_features:
            ## feature matching loss
            # shape of x_fake_ and x_real: (max_seq_len * bs, dim)
            x_fake_ = dout_fake['disc_features']
            x_real_ = dout_real['disc_features'].detach()

            # take a mean dim-wise and get an average over bs * seq after taking the loss
            if self.opts.seq_weights == True:
                # multiply 2.0 to get the same expected loss as the original loss (may erase it later)
                seq_weights = torch.tensor([2.0*(i+1)/self.opts.max_seq_len for i in range(self.opts.max_seq_len)]
                                      ).reshape(self.opts.max_seq_len, 1, 1).to(self.opts.device)
                # reshape loss_l1_disc_features from (max_seq_len * bs, dim) to (max_seq_len, bs, dim), then multiply it by
                # seq_weights in order to avoid OOM, which means we can reduce space complexity of seq_weights
                # from O(max_seq_len * bs) to O(max_seq_len)
                loss_l1_disc_features = self.feat_loss_criterion(x_fake_, x_real_, reduction='none'
                                                                 ).reshape(self.opts.max_seq_len, batch_size, -1) * seq_weights
                # reshape loss_l1_disc_features back from (max_seq_len, bs, dim) to (max_seq_len * bs, dim)
                loss_l1_disc_features = loss_l1_disc_features.reshape(self.opts.max_seq_len * batch_size, -1
                                                                      ).mean(1).sum(0) / x_fake_.size(0)
            else:
                loss_l1_disc_features = self.feat_loss_criterion(x_fake_, x_real_, 
                                                    reduction='none').mean(1).sum(0) / x_fake_.size(0)
            
            loss_dict['loss_l1_disc_features'] = loss_l1_disc_features
            total_loss += self.opts.feature_loss_multiplier * (loss_l1_disc_features)

        ## recon_loss
        recon_multiplier = self.opts.recon_loss_multiplier
        x_fake_ = gen_adv_input
        x_real_ = states[1:len(gen_adv_input)+1]

        loss_recon = self.get_recon_loss(x_fake_, x_real_, div=x_fake_.size(0))
        loss_dict['loss_recon'] = loss_recon
        total_loss += recon_multiplier * loss_recon

        if train:
            total_loss.backward()
            self.optG.step()

        return loss_dict, total_loss, gout, mems

    def discriminator_trainstep(self, states, actions, neg_actions, warm_up=10, gout=None, it=0):
        '''
        Run one step of discriminator

        shape of args:
            if self.opts.is_bs_first == True:
                states: (bs, seq_len + 1, feat)
                actions: [(bs, seq_len) * num_agents]
            else:
                states: (seq_len + 1, bs, feat)
                actions: [(seq_len, bs) * num_agents]
        '''
        batch_size = states.size(0) if self.opts.is_bs_first else states.size(1)

        if self.opts.warmup_decay_step > 0:
            warm_up = max(self.opts.min_warmup, math.ceil(warm_up * (1 - it * 1.0 / self.opts.warmup_decay_step)))

        utils.toggle_grad(self.netG, False)
        utils.toggle_grad(self.netD, True)
        self.netG.train()
        self.netD.train()

        self.optG.zero_grad()
        self.optD.zero_grad()

        loss_dict = {}
        #states = [x.requires_grad_() for x in states]
        #actions = [x.requires_grad_() for x in actions]
        #neg_actions = [x.requires_grad_() for x in neg_actions]

        ################# On real data ####################
        #d_input = torch.cat(states[1:], dim=0)
        d_input, states, actions, neg_actions = self.adjust_inputs_for_disc(states, states, actions, neg_actions,
                                                                                            self.opts.is_bs_first)
        d_input = d_input.requires_grad_()
        dout = self.netD(d_input, actions, states, warm_up, neg_actions)

        loss = 0
        dloss_real_content_action_wrong_loss = 0

        # single frame loss
        dloss_real_single_frame_loss = self.discriminator_loss(dout['single_frame_predictions_all'], 1)
        loss_dict['dloss_real_single_frame_loss'] = dloss_real_single_frame_loss

        # temporal loss - false actions
        if self.opts.temporal_hierarchy:
            for i in range(len(dout['neg_content_predictions'])):
                curloss = self.discriminator_loss(dout['neg_content_predictions'][i], 0)
                loss_dict['dloss_real_content_action_wrong_loss' + str(i)] = curloss
                dloss_real_content_action_wrong_loss += curloss
            dloss_real_content_action_wrong_loss = dloss_real_content_action_wrong_loss / len(dout['neg_content_predictions'])
            loss_dict['dloss_real_content_action_wrong_loss'] = dloss_real_content_action_wrong_loss

        # action reconstruction loss
        actions_recon_loss = self.calculate_action_recon_loss(dout, actions, batch_size)
        for i in range(len(actions_recon_loss)):
            loss_dict[f'd_action{i}_recon_loss'] = actions_recon_loss[i]

        # temporal loss - true actions
        dloss_real_content_loss = 0
        if self.opts.temporal_hierarchy:
            for i in range(len(dout['content_predictions'])):
                curloss = self.discriminator_loss(dout['content_predictions'][i], 1)
                loss_dict['dloss_real_content_loss' + str(i)] = curloss
                dloss_real_content_loss += curloss
            dloss_real_content_loss = dloss_real_content_loss / len(dout['content_predictions'])
            loss_dict['dloss_real_content_loss'] = dloss_real_content_loss


        loss += (dloss_real_content_loss + 0.2* dloss_real_content_action_wrong_loss) + \
                 dloss_real_single_frame_loss + sum(actions_recon_loss)

        # regularization
        reg = 0
        #d_input = d_input.reshape(self.opts.max_seq_len*batch_size, -1)
        if self.reg_param > 0:
            # 0.25 may need to be adjusted later
            reg_multiplier = 1 / (self.opts.num_agents + 2)
            reg += reg_multiplier*utils.compute_grad2(dout['single_frame_predictions_all'], d_input, ns=self.opts.max_seq_len).mean()
            for i in range(len(dout['action_recons'])):
                reg += reg_multiplier*utils.compute_grad2(dout['action_recons'][i], d_input, ns=self.opts.max_seq_len).mean()

            reg_temporal = 0
            if self.opts.temporal_hierarchy:
                for i in range(len(dout['content_predictions'])):
                    curloss = utils.compute_grad2(dout['content_predictions'][i], d_input, ns=self.opts.max_seq_len).mean()
                    reg_temporal += curloss
                reg_temporal = reg_temporal / len(dout['content_predictions'])
                loss_dict['dloss_REG_temporal'] = reg_temporal

            loss_dict['dloss_REG'] = reg
            loss += self.reg_param * reg + self.opts.LAMBDA_temporal * reg_temporal


        ################# On fake data ####################
        dout_fake = self.netD(gout.detach(), actions, states, warm_up)
        loss_dict, fake_loss = self.calculate_discriminator_adv_loss_fake(loss_dict, dout_fake)
        loss += fake_loss

        loss.backward()
        self.optD.step()
        utils.toggle_grad(self.netD, False)

        return loss_dict
