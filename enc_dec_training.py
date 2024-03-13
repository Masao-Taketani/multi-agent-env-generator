import torch
from torch import autograd, optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from torchvision import utils as tvis_utils
import os
import argparse

from models.enc_dec import EncoderDecoder
from models.discriminator import Discriminator
from data.encdec_datahandler import ImageDataset
from lpips import networks_basic as networks
from utils import sample_data, data_sampler, collate_fn


torch.backends.cudnn.benchmark = True


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def get_disc_loss(real_pred, fake_pred, loss_dict, args):
    ind = 0
    d_loss = 0
    for rp, fp in zip(real_pred, fake_pred):
        if args.gan_loss_type == 'logistic':
            cur_loss = d_logistic_loss(rp, fp)
        elif args.gan_loss_type == 'hinge':
            cur_loss = d_hinge_loss(rp, fp)
        else:
            raise ValueError(f'{args.gan_loss_type} is not supported')

        loss_dict['real_score'+str(ind)] = rp.detach().mean()
        loss_dict['fake_score'+str(ind)] = fp.detach().mean()

        d_loss += cur_loss
        ind += 1
    d_loss = d_loss / ind
    return d_loss

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def d_hinge_loss(real_pred, fake_pred):
    real_loss = torch.mean(F.relu(1. - real_pred))
    fake_loss = torch.mean(F.relu(1. + fake_pred))
    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.mean(1).sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def get_disc_r1_loss(real_pred, real_img, loss_dict):
    ind = 0
    r1_loss = 0
    for rp in real_pred:
        cur_loss = d_r1_loss(rp, real_img)
        loss_dict['r1_' + str(ind)] = cur_loss
        r1_loss += cur_loss
        ind += 1
    r1_loss = r1_loss / ind
    return r1_loss

def get_g_loss(fake_pred, loss_dict):
    ind = 0
    g_loss = 0
    for fp in fake_pred:
        if args.gan_loss_type == 'logistic':
            cur_loss = g_logistic_loss(fp)
        elif args.gan_loss_type == 'hinge':
            cur_loss = g_hinge_loss(fp)
        else:
            raise ValueError(f'{args.gan_loss_type} is not supported')
        
        loss_dict['g' + str(ind)] = cur_loss
        g_loss += cur_loss
        ind += 1
    g_loss = g_loss / ind
    loss_dict['g'] = g_loss
    return g_loss

def g_logistic_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def g_hinge_loss(fake_pred):
    loss = -torch.mean(fake_pred)
    return loss

def get_pixelwise_loss(fake_img, real_img, loss_dict, args):
    if args.pixelwise_norm == 'l1':
        pixelwise_loss = F.l1_loss(fake_img, real_img, reduction=args.pixelwise_loss_reduction)
    elif args.pixelwise_norm == 'l2':
        pixelwise_loss = F.mse_loss(fake_img, real_img, reduction=args.pixelwise_loss_reduction)
    else:
        raise ValueError(f'{args.pixelwise_norm} is not supported')

    if loss_dict is not None:
        loss_dict['pixelwise_loss'] = pixelwise_loss
    return pixelwise_loss

def get_perceptual_loss(fake_img, real_img, loss_dict, percept):
    if percept is not None:
        perceptual_loss = percept(fake_img, real_img).mean()
    else:
        assert 0, 'percept has not to be None if you want to use perceptual_loss'
    if loss_dict is not None:
        loss_dict['perceptual_loss'] = perceptual_loss
    return perceptual_loss

def get_data(loader, device):
    real_img = next(loader)
    real_img = real_img.to(device)
    return real_img

def save_img(name, data, n_sample, logger, step, scale=True):
    sample = data[:n_sample]
    if scale:
        # scale from [-1, 1] t0 [0, 1]
        sample = sample * 0.5 + 0.5
    sample = torch.clamp(sample, 0, 1.0)
    x = tvis_utils.make_grid(
        sample, nrow=int(n_sample ** 0.5),
        normalize=False, scale_each=False
    )
    logger.add_image(name, x, step)

def validation(encdec, logger, val_loader, args, idx, percept):
    print('[Start validation]')
    encdec.eval()
    discriminator.eval()
    val_losses = {}
    if args.use_pixelwise_loss:
        val_losses['val_pixelwise_loss'] = 0
    if args.use_perceptual_loss:
        val_losses['val_perceptual_loss'] = 0
    
    # Set 30 as the number of validation iterations to save time
    num_val = 30
    for ind_val in range(num_val):
        real_img = get_data(val_loader, device)
        out = encdec(real_img)
        recon_img = out
        if args.use_pixelwise_loss:
            pixelwise_loss = get_pixelwise_loss(recon_img, real_img, None, args)
            val_losses['val_pixelwise_loss'] += pixelwise_loss.data.item()
        if args.use_perceptual_loss:
            perceptual_loss = get_perceptual_loss(recon_img, real_img, None, percept)
            val_losses['val_perceptual_loss'] += perceptual_loss.data.item()
        #if ind_val % max(1, num_val // 10) == 0:
        if ind_val % 10 == 0:
            save_img('VAL_Img/recon_img', recon_img, args.n_sample, logger, idx+ind_val)
            save_img('VAL_Img/real_img', real_img, args.n_sample, logger, idx+ind_val)
            print(str(ind_val)+'/'+str(num_val))
        del out, real_img

    for key, val in val_losses.items():
        logger.add_scalar('VAL_Scalar/'+key, val / num_val, idx)

def train_step(encdec, discriminator, encdec_optim, d_optim, logger, loader, args, i, percept):
    '''
    run one step of training
    '''
    loss_dict = {}

    encdec.train()
    discriminator.train()

    if i > args.iter:
        print('Done!')
        exit(-1)

    real_img = get_data(loader, device)

    ######################### DISCRIMINATOR STEP #########################
    requires_grad(encdec, False)
    requires_grad(discriminator, True)

    # run encdec and discriminator
    out = encdec(real_img)
    fake_pred = discriminator(out)
    real_pred = discriminator(real_img)
    d_loss = get_disc_loss(real_pred, fake_pred, loss_dict, args)

    d_loss /= args.accum_iter
    d_loss.backward()
    if (i+1) % args.accum_iter == 0:
        d_optim.step()
        discriminator.zero_grad()

    # regularization
    d_regularize = i % (args.d_reg_every * args.accum_iter) == 0
    if d_regularize:
        real_img.requires_grad = True
        real_pred = discriminator(real_img)
        r1_loss = get_disc_r1_loss(real_pred, real_img, loss_dict)

        discriminator.zero_grad()

        (args.r1 / 2 * r1_loss * args.d_reg_every).backward()
        d_optim.step()
        discriminator.zero_grad()
        loss_dict['r1'] = r1_loss


    ######################### ENC DEC STEP #########################
    requires_grad(encdec, True)
    requires_grad(discriminator, False)

    ## run encdec and discriminator
    out = encdec(real_img)
    recon_img = out
    fake_pred = discriminator(recon_img)

    ## losses
    g_loss = get_g_loss(fake_pred, loss_dict)
    if args.use_pixelwise_loss:
        pixelwise_loss = get_pixelwise_loss(recon_img, real_img, loss_dict, args)
        g_loss += args.pixel_coef * pixelwise_loss
    if args.use_perceptual_loss:
        perceptual_loss = get_perceptual_loss(recon_img, real_img, loss_dict, percept)
        g_loss += args.percept_coef * perceptual_loss
    g_loss /= args.accum_iter
    g_loss.backward()
    if (i+1) % args.accum_iter == 0:
        encdec_optim.step()
        encdec.zero_grad()

    ######################### LOGGING #########################
    if i % args.log_loss_iter == 0:
        # log losses
        loss_str = 'step '+str(i)+': '
        for key, val in loss_dict.items():
            loss_dict[key] = val.mean().item()
            loss_str += key + '=' + str(loss_dict[key])[:5] + ', '
            logger.add_scalar('Scalar/' + key, loss_dict[key], i)
        print(loss_str)

    if i % args.log_img_iter == 0:
        encdec.eval()
        with torch.no_grad():
            ## log images
            save_img('Img/recon_img', recon_img, args.n_sample, logger, i)
            save_img('Img/real_img', real_img, args.n_sample, logger, i)

    del out

def train(args, loader, val_loader, encdec, discriminator, encdec_optim, d_optim, logger, percept=None):
    ### need to check later
    loader = sample_data(loader)
    val_loader = sample_data(val_loader)

    for idx in range(args.iter):
        if idx % args.val_iter == 0:
            with torch.no_grad():
                validation(encdec, logger, val_loader, args, idx, percept)

        i = idx + args.start_iter
        train_step(encdec, discriminator, encdec_optim, d_optim, logger, loader, args, i, percept)

        if i % args.save_iter == 0 and i > 0:
            # save model
            save_dict = {
                'encdec': encdec.state_dict(),
                'encdec_optim': encdec_optim.state_dict(),
                'args': args
            }
            save_dict['d'] =  discriminator.state_dict()
            save_dict['d_optim'] = d_optim.state_dict()
            torch.save(save_dict, os.path.join(args.log_dir, str(i)+'.pt'))
            del save_dict


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--iter', type=int, default=800000)
    parser.add_argument('--log_loss_iter', type=int, default=25)
    parser.add_argument('--log_img_iter', type=int, default=1000)
    parser.add_argument('--save_iter', type=int, default=10000)
    parser.add_argument('--val_iter', type=int, default=1000)

    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--n_sample', type=int, default=6)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--gan_loss_type', type=str, default='hinge')
    parser.add_argument('--use_pixelwise_loss', action='store_true')
    parser.add_argument('--pixel_coef', type=float, default=50.0)
    parser.add_argument('--pixelwise_norm', type=str, default='l1')
    parser.add_argument('--pixelwise_loss_reduction', type=str, default='mean')
    parser.add_argument('--use_perceptual_loss', action='store_true')
    parser.add_argument('--percept_coef', type=float, default=50.0)

    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--ckpt', type=str, default=None)

    parser.add_argument('--log_dir', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dataset', type=str, default='pong')
    parser.add_argument('--num_patchD', type=int, default=0)

    parser.add_argument('--img_size', type=str, default='64x64', help='heightxwidth')
    parser.add_argument('--in_channel', type=int, default=3)
    parser.add_argument('--width_mul', type=float, default=1)
    parser.add_argument('--crop_input', type=int, default=0)
    parser.add_argument('--d_base_channel', type=int, default=512)
    parser.add_argument('--enc_base_channel', type=int, default=512)
    parser.add_argument('--dec_hidden_dim', type=int, default=512)
    parser.add_argument('--no_in', action='store_true', default=True)
    parser.add_argument('--nfilterDec', type=int, default=32)

    parser.add_argument('--accum_iter', type=int, default=1, 
                        help='specify number of gradient accumulation')

    args = parser.parse_args()
    args.start_iter = 0

    args.img_size = tuple([int(i) for i in args.img_size.split('x')])

    percept = None
    if args.use_perceptual_loss:
        # net for perceptual loss
        percept = networks.PNetLin(pnet_rand=False, pnet_tune=False, pnet_type='vgg',
                                   use_dropout=True, spatial=False, version='0.1', lpips=True).to(device)
        model_path = './lpips/weights/v0.1/vgg.pth'
        print('Loading vgg model from: %s' % model_path)
        percept.load_state_dict(torch.load(model_path), strict=False)

    # init models
    encdec = EncoderDecoder(args).to(device)
    discriminator = Discriminator(args).to(device)

    # optimizers
    encdec_optim = optim.Adam(
        encdec.parameters(),
        lr=args.lr
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr
    )

    # load ckpt if continuing training
    if args.ckpt is not None:
        print('load model:', args.ckpt)
        ckpt = torch.load(args.ckpt, map_location='cpu')

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass

        encdec.load_state_dict(ckpt['encdec'], strict=True)
        discriminator.load_state_dict(ckpt['d'], strict=True)
        encdec_optim.load_state_dict(ckpt['encdec_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
        del ckpt

    # load training and validation datasets
    img_dataset = ImageDataset(args.data_path, args.img_size, args.dataset, train=True, args=args)
    loader = data.DataLoader(
        img_dataset,
        batch_size=args.batch,
        sampler=data_sampler(img_dataset, shuffle=True),
        drop_last=True,
        collate_fn=collate_fn
    )
    print('Total training dataset length: ' + str(len(img_dataset)))

    val_dataset = ImageDataset(args.data_path, args.img_size, args.dataset, train=False, args=args)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch,
        sampler=data_sampler(val_dataset, shuffle=False),
        drop_last=True,
        collate_fn=collate_fn
    )
    print('Total validation dataset length: ' + str(len(val_dataset)))

    os.makedirs(args.log_dir, exist_ok=True)
    logger = SummaryWriter(args.log_dir)
    
    train(args, loader, val_loader, encdec, discriminator, encdec_optim, d_optim, logger, percept)