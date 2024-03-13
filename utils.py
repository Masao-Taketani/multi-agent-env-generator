import numpy as np
from PIL import Image
import torch
from torch import autograd
from torch.autograd import Variable
from torch.utils import data
from termcolor import colored


def create_gif_from_npz(fpath, gif_name):
    data = np.load(fpath, allow_pickle=True)
    np_imgs = data['states']
    images = []
    for img in np_imgs:
        images.append(Image.fromarray(img))
    #print(data['actions'])
    images[0].save(gif_name, save_all=True, append_images=images[1:], 
                   optimize=False, duration=20, loop=0)

def check_arg(opts, arg):
    v = vars(opts)
    if arg in v:
        if type(v[arg]) == bool:
            return v[arg]
        else:
            return True
    else:
        return False
    
def list_to_dict(l):
    d = {}
    for entry in l:
        d[entry] = 1
    return d

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def get_data(opts, device, states, actions, neg_actions=None):
    rtn_acts = []
    if opts.use_neg_actions == True: rtn_neg_acts = []
    
    states = torch.cat(states, axis=1).to(device)
    for i in range(opts.num_agents):
        rtn_acts.append(torch.cat(actions[i], axis=1).to(device))
        if opts.use_neg_actions == True: rtn_neg_acts.append(torch.cat(neg_actions[i], 
                                                                       axis=1).to(device))
    return states, rtn_acts, rtn_neg_acts

def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)
    
def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""
def compute_grad2(d_out, x_in, allow_unused=False, batch_size=None, gpu=0, ns=1):
    if d_out is None:
        return check_gpu(gpu, torch.FloatTensor([0]))
    if batch_size is None:
        batch_size = x_in.size(0)

    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True,
        allow_unused=allow_unused
    )[0]
    # import pdb; pdb.set_trace();

    grad_dout2 = grad_dout.pow(2)
    # xassert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1) * (ns * 1.0 / 6)
    return reg

def check_gpu(gpu, *args):
    '''
    '''
    if gpu == None or gpu < 0:
        if isinstance(args[0], dict):
            d = args[0]
            var_dict = {}
            for key in d:
                var_dict[key] = Variable(d[key])
            if len(args) > 1:
                return [var_dict] + check_gpu(gpu, *args[1:])
            else:
                return [var_dict]
        if isinstance(args[0], list):
            return [Variable(a) for a in args[0]]
        # a list of arguments
        if len(args) > 1:
            return [Variable(a) for a in args]
        else:
            return Variable(args[0])

    else:
        if isinstance(args[0], dict):
            d = args[0]
            var_dict = {}
            for key in d:
                var_dict[key] = Variable(d[key]).to('cuda')
            if len(args) > 1:
                return [var_dict] + check_gpu(gpu, *args[1:])
            else:
                return [var_dict]
        if isinstance(args[0], list):
            return [Variable(a).to('cuda') for a in args[0]]
        # a list of arguments
        if len(args) > 1:
            return [Variable(a).to('cuda') for a in args]
        else:
            return Variable(args[0]).to('cuda')

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def print_color(txt, color):
    ''' print <txt> to terminal using colors
    '''
    print(colored(txt, color))
    return

def save_model(fname, epoch, netG, netD, opts):
    outdict = {'epoch': epoch, 'translearner': netG.state_dict(), 
               'disc': netD.state_dict(), 'opts': opts}
    torch.save(outdict, fname)

def save_optim(fname, epoch, optG, optD):
    outdict = {'epoch': epoch, 'tl_opt': optG.state_dict(), 'disc_opt': optD.state_dict()}
    torch.save(outdict, fname)