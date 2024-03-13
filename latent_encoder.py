import torch
from torch.utils import data
from torchvision import utils as tvis_utils
import argparse
import os
import numpy as np
import json

from models.enc_dec import EncoderDecoder
from data.encdec_datahandler import EpisodeDataset
from utils import sample_data


def load_data(ld, device, args):
    next_data = next(ld)
    if next_data is None:
        return None

    imgs, path, key = next_data
    imgs = imgs.to(device)
    imgs = imgs.squeeze(0)
    # path[0] means a key is taken from the list
    path = path[0] if not 'pilotnet' in args.dataset else path
    return imgs, path, key

def make_image(tensor, logger, title, video=True, global_step=0):
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1.0)

    if not video:
        tensor = tvis_utils.make_grid(
            tensor, nrow=1,
            normalize=False, scale_each=False
        )
        logger.add_image(title, tensor, global_step=global_step)
    else:
        logger.add_video(title, tensor.unsqueeze(0), global_step=global_step)

def visualize(logger, g_ema, imgs, merged_results, path, args, cur_fname=None):
    if isinstance(path, list):
        p = path[0][0]
    if isinstance(path, tuple):
        p = path[0]
    cur_dir = p.split('/')[-2]
    cur_fname = p.split('/')[-1] if cur_fname is None else cur_fname
    make_image(imgs, logger, cur_dir+'_'+cur_fname+'/GT')

    bs = merged_results['theme_mu'].size(0)
    #minibatch_size = bs # set smaller if gpu mem exceeds
    minibatch_size = args.vis_bs
    collect = []
    for mind in range(int(np.ceil(bs / minibatch_size))):
        cur_gt = imgs[mind*minibatch_size:(mind+1)*minibatch_size]
        cur_theme = merged_results['theme_mu'][mind*minibatch_size:(mind+1)*minibatch_size]
        cur_spatial = merged_results['spatial_mu'][mind*minibatch_size:(mind+1)*minibatch_size].view(minibatch_size, -1, g_ema.constant_input_size, int(g_ema.constant_input_size*g_ema.args.width_mul))

        out = g_ema({'theme_z':cur_theme, 'spatial_z': cur_spatial}, decode_only=True)
        collect.append(out['image'])
    out = {'image': torch.cat(collect, dim=0)}
    make_image(out['image'], logger, cur_dir+'_'+cur_fname+'/z')

def visualize_ma(logger, decoder, imgs, merged_results, path, args, cur_fname=None):
    if isinstance(path, list):
        p = path[0][0]
    if isinstance(path, tuple):
        p = path[0]
    cur_dir = p.split('/')[-2]
    cur_fname = p.split('/')[-1] if cur_fname is None else cur_fname
    make_image(imgs, logger, cur_dir+'_'+cur_fname+'/GT')

    bs = merged_results['latent_imgs'].size(0)
    #minibatch_size = bs # set smaller if gpu mem exceeds
    minibatch_size = args.vis_bs
    collect = []
    for mind in range(int(np.ceil(bs / minibatch_size))):
        cur_latent_imgs = merged_results['latent_imgs'][mind*minibatch_size:(mind+1)*minibatch_size]

        out = decoder(cur_latent_imgs)
        collect.append(out)
    out = {'image': torch.cat(collect, dim=0)}
    make_image(out['image'], logger, cur_dir+'_'+cur_fname+'/z')

# To save multi-agent data
def save_ma_data(merged_results, path, key, args, compression_type='npz'):
    if isinstance(path, tuple):
        path = path[0]
        path = os.path.dirname(path)
    npz_path = os.path.join(path, 'action_log.npz')

    if compression_type == 'npz':
        cur_file = {}
        cur_file['actions'] = np.load(npz_path, allow_pickle=True)['actions']
    else:
        print(f'compression type {compression_type} is not supported')
        exit(-1)

    # this process is temporal. Planning to modify it later
    path = [p[0] for p in path]

    for k, item in merged_results.items():
        cur_file[k] = item.cpu().data.numpy()

    cur_file['paths'] = path
    cur_fname = os.path.join(args.results_path, key[0]+'.npy')
    np.save(cur_fname, cur_file)

def save_data(merged_results, ind, path, key, args):
    if isinstance(path, tuple):
        path = path[0]
        path = os.path.dirname(path)
    json_path = os.path.join(path, 'info.json')

    cur_dir = path.split('/')[-2]
    cur_fname = path.split('/')[-1]

    cur_file = json.load(open(json_path, 'rb'))

    path = [p[0] for p in path]

    for k, item in merged_results.items():
        cur_file[k] = item.cpu().data.numpy()

    cur_file['paths'] = path
    cur_fname = os.path.join(args.results_path, key[0]+'.npy')
    np.save(cur_fname, cur_file)

def encode(encdec, loader, args, device):
    num_batch = len(loader)
    print('\n\nnum_batch: ' + str(num_batch) + '\n\n')

    loader = sample_data(loader)

    logger = None
    if args.visualize:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(args.results_path)

    for ind in range(num_batch):
        next_data = load_data(loader, device, args)
        if next_data is None:
            continue
        imgs, path, key = next_data
        if 'carla' in args.dataset or 'gibson' in args.dataset or 'pong' in args.dataset \
            or 'boxing' in args.dataset or 'quadrapong' in args.dataset or 'gtav' in args.dataset:
            cur_fname = os.path.join(args.results_path, key[0] + '.npy')
        else:
            print('dataset type not implemented')
            exit(-1)
        
        if os.path.exists(cur_fname):
            continue
        
        partial_batch_size = imgs.size(0) // args.num_div_batch
        rest = imgs.size(0) - partial_batch_size * args.num_div_batch
        outputs = []
        for div_ind in range(args.num_div_batch):
            with torch.no_grad():
                cur_imgs = imgs[div_ind * partial_batch_size: (div_ind+1) * partial_batch_size]
                res = encdec.enc(cur_imgs)
                outputs.append(res)
                continue
        if rest > 0:
            with torch.no_grad():
                cur_imgs = imgs[(div_ind+1) * partial_batch_size:]
                res = encdec.enc(cur_imgs)
                outputs.append(res)

        merged_results = {}
        if 'pong' in args.dataset or 'boxing' in args.dataset or 'quadrapong' in args.dataset \
            or 'gtav' in args.dataset or 'carla' in args.dataset:
            merged_results['latent_imgs'] = []
            for out in outputs:
                merged_results['latent_imgs'].append(out)
            merged_results['latent_imgs'] = torch.cat(merged_results['latent_imgs'], dim=0)
        elif 'gibson' in args.dataset:
            keys = list(outputs[0].keys())
            for k in keys:
                if outputs[0][k] is not None and not 'loss' in k:
                    merged_results[k] = []
                    for out in outputs:
                        merged_results[k].append(out[k])
                    merged_results[k] = torch.cat(merged_results[k], dim=0)
        else:
            print('dataset type not implemented')
            exit(-1)

        with torch.no_grad():
            if args.visualize:
                ## for checking purpose
                if 'pong' in args.dataset or 'boxing' in args.dataset or 'quadrapong' in args.dataset \
                    or 'gtav' in args.dataset or 'carla' in args.dataset:
                    visualize_ma(logger, encdec.dec, imgs, merged_results, path, args, cur_fname=str(ind))
                elif 'gibson' in args.dataset:
                    visualize(logger, encdec.dec, imgs, merged_results, path, args, cur_fname=str(ind))
                else:
                    print('dataset type not implemented')
                    exit(-1)

            if 'pong' in args.dataset or 'boxing' in args.dataset or 'quadrapong' in args.dataset \
               or 'gtav' in args.dataset or 'carla' in args.dataset:
                save_ma_data(merged_results, path, key, args)
            elif 'gibson' in args.dataset:
                save_data(merged_results, str(args.cur_ind) + '_' + str(ind), path, key,args)
            else:
                print('dataset type not implemented')
                exit(-1)
            print('%d/%d' % (ind, num_batch))


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--img_size', type=str, default='64x64', help='heightxwidth')
    parser.add_argument('--cur_ind', type=int, default=0)

    parser.add_argument('--num_chunk', type=int, default=0)
    parser.add_argument('--num_div_batch', type=int, default=120)

    parser.add_argument('--results_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='pong')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vis_bs', type=int, default=16)
    parser.add_argument('--width_mul', type=float, default=1.0)
    parser.add_argument('--crop_input', type=int, default=0)
    parser.add_argument('--test', type=int, default=0)
    args = parser.parse_args()

    args.img_size = tuple([int(i) for i in args.img_size.split('x')])

    #### load trained EncoderDecoder model
    saved_file = torch.load(args.ckpt)
    saved_args = saved_file['args']
    encdec = EncoderDecoder(saved_args).to(device)
    encdec.load_state_dict(saved_file['encdec'], strict=False)

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    #### define dataloader
    dataset = EpisodeDataset(args.data_path, args.img_size, args.dataset, args=args)
    loader = data.DataLoader(
        dataset,
        # batch_size 1 indicates the loader returns one episode data each time
        batch_size=1,
        sampler=data.SequentialSampler(dataset),
        drop_last=False,
        collate_fn=collate_fn
    )

    encode(encdec, loader, args, device)