"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

import os
import torch
import time
import argparse
import utils
from trainer import Trainer
#import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import copy

from models.transformer import TransitionLearner
from models.discriminator_translearner import Discriminator
from data.translearner_datahandler import get_custom_dataset


def train_translearner(gpu, opts):
    torch.backends.cudnn.benchmark = True

    opts = copy.deepcopy(opts)
    opts.gpu = gpu
    opts.input_detach = True
    opts.do_latent = True
    opts.temporal_hierarchy = True
    opts.disc_features = True
    opts.warmup_decay_step = 0
    opts.use_neg_actions = True
    warm_up = None

    torch.manual_seed(opts.seed)

    # dataset ---
    print('setting up dataset')

    #train_loaders = dataloader.get_custom_dataset(opts, set_type=0, getLoader=True, num_workers=4)
    #val_loaders = dataloader.get_custom_dataset(opts, set_type=1, getLoader=True, num_workers=4)
    train_loader, train_len = get_custom_dataset(opts, train=True)
    val_loader, val_len = get_custom_dataset(opts, train=False)
    print('train total iters:', train_len)
    print('validation total iters:', val_len)
    ## create model
    #netG, netD = utils.build_models(opts)
    ## choose optimizer
    #optD = utils.choose_optimizer(netD, opts, opts.lrD)

    #keyword = 'graphic'
    #optG_temporal = utils.choose_optimizer(netG, opts, opts.lrG_temporal, exclude=+keyword,
    #                                       model_name='optG_temporal')
    #optG_graphic = utils.choose_optimizer(netG, opts, opts.lrG_graphic, include=keyword, model_name='optG_graphic')

    # init models
    translearner = TransitionLearner(opts).to(device)
    disc = Discriminator(opts).to(device)

    # optimizers
    tl_optim = torch.optim.Adam(translearner.parameters(), lr=opts.G_lr, betas=(0.0, 0.9))
    disc_optim = torch.optim.Adam(disc.parameters(), lr=opts.D_lr, betas=(0.0, 0.9))

    logging = True
    if logging:
        os.makedirs(opts.log_dir, exist_ok=True)
        logger = SummaryWriter(opts.log_dir)

    trainer = Trainer(opts, translearner, disc, tl_optim, disc_optim, opts.LAMBDA)

    it = opts.start_iter
    print('Start iteration %d...' % it) if logging else None
    break_while = False
    while True:
        torch.cuda.empty_cache()
        #times = []
        for states, actions, neg_actions in train_loader:
            it += 1
            start = time.time()
            # prepare data
            states, actions, neg_actions = utils.get_data(opts,
                                                          device, 
                                                          states, 
                                                          actions, 
                                                          neg_actions)
            utils.print_color('Data loading:%f' % (time.time()-start), 'yellow')

            # Generators updates
            start = time.time()

            gloss_dict, gloss, gout, _ = trainer.generator_trainstep(states, actions, warm_up, it=it)
            gtime = time.time() - start

            # Discriminator updates
            start1 = time.time()
            dloss_dict = trainer.discriminator_trainstep(states, actions, neg_actions, warm_up, gout, it=0)
            dtime = time.time() - start1

            # Log
            if logging:
                with torch.no_grad():

                    loss_str = '[Generator step %d] ' % (it)
                    for k, v in gloss_dict.items():
                        if not (type(v) is float):
                            if (it % opts.log_iter) == 0:
                                logger.add_scalar('losses/' + k, v.data.item(), it)
                            loss_str += k + ': ' + str(v.data.item())[:5] + ', '
                    print(loss_str)
                    utils.print_color('netG update:%f' % (gtime), 'yellow')

                    #if step % 1000  == 0 and epoch %  max(1, opts.eval_epoch) == 0:
                    #    visual_utils.draw_output(gout, actions, neg_actions, states, opts, vutils,
                    #                             logger,
                    #                             it, latent_decoder=latent_decoder,
                    #                             tag='trn_images')

                    loss_str = '[Discriminator step %d] ' % (it)
                    for k, v in dloss_dict.items():
                        if not type(v) is float:
                            if (it % opts.log_iter == 0):
                                logger.add_scalar('losses/' + k, v.data.item(), it)
                            loss_str += k + ': ' + str(v.data.item())[:5] + ', '
                    print(loss_str)
                    utils.print_color('netD update:%f' % (dtime), 'yellow')
            del gloss_dict, gloss, gout, states, actions, neg_actions, dloss_dict
            torch.cuda.synchronize()
            #times += [time.time() - start]
            #print(f"Average iteration time: {np.mean(times)}")

            if it % opts.eval_iter == 0:
                print('Validation iteration %d...' % it) if logging else None
                torch.cuda.empty_cache()

                val_losses = {}
                trainer.netG.eval()
                num_val = 30
                val_it = 0
                while True:
                    if val_it == num_val:
                        break

                    for states, actions, neg_actions in val_loader:
                        # prepare data
                        states, actions, neg_actions = utils.get_data(opts,
                                                                      device, 
                                                                      states, 
                                                                      actions, 
                                                                      neg_actions)
                        
                        with torch.no_grad():
                            loss_dict, gloss, gout, _ = trainer.generator_trainstep(states, actions, warm_up, 
                                                                                    train=False, it=it)
                            if logging:
                                for key, val in loss_dict.items():
                                    if key in val_losses:
                                        val_losses[key] += val.item()
                                    else:
                                        val_losses[key] = val.item()
                                #if step % vis_step == 0:
                                #    visual_utils.draw_output(gout, actions, neg_actions, states, opts, vutils, logger, it,
                                #                      latent_decoder=latent_decoder, tag='val_images')
                            del loss_dict, gloss, gout, states, actions, neg_actions

                        if val_it % 10 == 0:
                            print(str(val_it)+'/'+str(num_val))

                    val_it += 1

                for key, val in val_losses.items():
                    logger.add_scalar('val_losses/'+key, val / val_len, it)

            if it % opts.save_iter == 0 and logging:
                print('Saving checkpoint')
                utils.save_model(os.path.join(opts.log_dir, 'model' + str(it) + '.pt'), 
                                it, translearner, disc, opts)
                utils.save_optim(os.path.join(opts.log_dir, 'optim' + str(it) + '.pt'), 
                                it, tl_optim, disc_optim)

            if it == opts.num_iters:
                break_while = True
                print('training is done!')
                break
        
        if break_while:
            break

    print('Saving for the last checkpoint')
    utils.save_model(os.path.join(opts.log_dir, 'model' + str(it) + '.pt'), 
                        it, translearner, disc, opts)
    utils.save_optim(os.path.join(opts.log_dir, 'optim' + str(it) + '.pt'), 
                        it, tl_optim, disc_optim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_iters', type=int, default=800000)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--save_iter', type=int, default=5000)
    parser.add_argument('--eval_iter', type=int, default=2500)
    parser.add_argument('--log_iter', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--LAMBDA', type=float, default=1.0)
    parser.add_argument('--LAMBDA_temporal', type=float, default=1.0)
    parser.add_argument('--feature_loss_multiplier', type=float, default=10.0)
    parser.add_argument('--recon_loss_multiplier', type=float, default=0.1)

    #parser.add_argument('--n_sample', type=int, default=6)

    parser.add_argument('--log_dir', type=str, default='translearner_results')
    parser.add_argument('--G_lr', type=float, default=0.0001)
    parser.add_argument('--D_lr', type=float, default=0.0001)
    parser.add_argument('--dataset', type=str, default='pong')
    parser.add_argument('--data_dir', type=str, default='encoded_dataset/pong/')
    parser.add_argument('--is_bs_first', type=int, default=1, 
                        help='whether data starts with (bs, seq) or (seq, bs). \
                        0 for False and 1 for True')

    parser.add_argument('--width_mul', type=float, default=1)
    parser.add_argument('--crop_input', type=int, default=0)
    parser.add_argument('--action_space', type=int, default=4)
    parser.add_argument('--continuous_action', action='store_true', default=False)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--act_drop_ratio', type=float, default=0.0)
    parser.add_argument('--tranenclayer_drop_ratio', type=float, default=0.1)
    parser.add_argument('--transinput_drop_ratio', type=float, default=0.0)
    parser.add_argument('--vis_droput_ratio', type=float, default=0.0)
    parser.add_argument('--transenc_heads', type=int, default=8)
    parser.add_argument('--num_transenclayer', type=int, default=2)
    parser.add_argument('--max_seq_len', type=int, default=64)
    parser.add_argument('--num_action_history', type=int, default=1)
    parser.add_argument('--pos_enc_type', type=str, default='original')
    parser.add_argument('--attn_mask_type', type=str, default='1st',
                        help='specify which type of attention to use')

    parser.add_argument('--test', action='store_true')
    parser.add_argument('--feat_loss', type=str, default='l1')
    parser.add_argument('--recon_loss', type=str, default='l2')

    parser.add_argument('--nfilterD', type=int, default=64)
    parser.add_argument('--latent_z_size', type=int, default=512)
    parser.add_argument('--nfilterD_temp', type=int, default=32)
    parser.add_argument('--config_temporal', type=int, default=64)
    parser.add_argument('--gen_content_loss_multiplier', type=float, default=1.5)
    parser.add_argument('--num_agents', type=int, default=2)
    parser.add_argument('--seq_weights', action='store_true', help='use this if you want to sequentially weigh \
                         recon loss and feat loss.')

    opts = parser.parse_args()
    opts.trans_type = 'vanilla'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    opts.device = device
    train_translearner(opts.gpu, opts)