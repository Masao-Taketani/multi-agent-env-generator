from torch import nn
from torch.nn import functional as F

try:
    from models.modules import ResBlock, View
except:
    from modules import ResBlock, View


def choose_global_netD(args):
    '''
    Global Discriminator
    '''
    if args.img_size[0] == 64:
        global_disc = nn.Sequential(
            nn.Conv2d(args.in_channel, args.d_base_channel // 4, 3, 1, 1), # 64x64x128
            nn.LeakyReLU(0.2),
            ResBlock(args.d_base_channel // 4, args.d_base_channel // 2), # 32x32x256
            ResBlock(args.d_base_channel // 2, args.d_base_channel), # 16x16x512
            ResBlock(args.d_base_channel, args.d_base_channel), # 8x8x512
            ResBlock(args.d_base_channel, 156), # 4x4x156
            ResBlock(156, args.d_base_channel), # 2x2x512
            ResBlock(args.d_base_channel, args.d_base_channel), # 1x1x512
            View((-1, args.d_base_channel)), # 512
            nn.Linear(args.d_base_channel, args.d_base_channel), # 512
            nn.LeakyReLU(0.2),
            nn.Linear(args.d_base_channel, 1), # scalar
        )
    elif args.img_size[0] == 48 and args.img_size[1] == 80:
        global_disc = nn.Sequential(
            nn.Conv2d(args.in_channel, args.d_base_channel // 4, 3, 1, 1), # 48x80x128
            nn.LeakyReLU(0.2),
            ResBlock(args.d_base_channel // 4, args.d_base_channel // 2), # 24x40x256
            ResBlock(args.d_base_channel // 2, args.d_base_channel), # 12x20x512
            ResBlock(args.d_base_channel, args.d_base_channel), # 6x10x512
            ResBlock(args.d_base_channel, 156), # 3x5x156
            #ResBlock(156, args.d_base_channel, downsample=False), # 3x5x512
            #ResBlock(args.d_base_channel, args.d_base_channel, downsample=False), # 3x5x512
            ResBlock(156, args.d_base_channel), # 2x3x512
            ResBlock(args.d_base_channel, args.d_base_channel), # 1x2x512
            ResBlock(args.d_base_channel, args.d_base_channel), # 1x1x512
            View((-1, args.d_base_channel)), # 512
            nn.Linear(args.d_base_channel, args.d_base_channel), # 512
            nn.LeakyReLU(0.2),
            nn.Linear(args.d_base_channel, 1), # scalar
        )
    elif args.img_size[0] == 256:
        global_disc = nn.Sequential(
            nn.Conv2d(args.in_channel, args.d_base_channel // 16, 3, 1, 1), # 256x256x32
            nn.LeakyReLU(0.2),
            ResBlock(args.d_base_channel // 16, args.d_base_channel // 8), # 128x128x64
            ResBlock(args.d_base_channel // 8, args.d_base_channel // 4), # 64x64x128
            ResBlock(args.d_base_channel // 4, args.d_base_channel // 2), # 32x32x256
            ResBlock(args.d_base_channel // 2, args.d_base_channel), # 16x16x512
            ResBlock(args.d_base_channel, args.d_base_channel), # 8x8x512
            ResBlock(args.d_base_channel, 156), # 4x4x156
            ResBlock(156, args.d_base_channel), # 2x2x512
            ResBlock(args.d_base_channel, args.d_base_channel), # 1x1x512
            View((-1, args.d_base_channel)), # 512
            nn.Linear(args.d_base_channel, args.d_base_channel), # 512
            nn.LeakyReLU(0.2),
            nn.Linear(args.d_base_channel, 1), # scalar
        )
    else:
        print(args.img_size)
        assert 0, 'model-%s not supported'

    return global_disc


def choose_patch_netD(args):
    '''
    Patch Discriminator
    '''
    if args.img_size[0] == 64 or args.img_size[0] == 48 or args.img_size[0] == 256:
        patch_disc = nn.Sequential(
            nn.Conv2d(args.in_channel, args.d_base_channel // 4, 3, 1, 1), # 64x64x128
            nn.LeakyReLU(0.2),
            ResBlock(args.d_base_channel // 4, args.d_base_channel // 2), # 32x32x256
            ResBlock(args.d_base_channel // 2, args.d_base_channel), # 16x16x512
            ResBlock(args.d_base_channel, args.d_base_channel), # 8x8x512
            ResBlock(args.d_base_channel, args.d_base_channel), # 4x4x512
            nn.Conv2d(args.d_base_channel, 1, 3, 1, 1), # 4x4x1
        )
    else:
        print(args.img_size)
        assert 0, 'model-%s not supported'

    return patch_disc


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.global_d = choose_global_netD(args)
        self.Ds = nn.ModuleList()
        for _ in range(args.num_patchD):
            self.Ds.append(choose_patch_netD(args))
    
    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, img):
        results = []

        results.append(self.global_d(img))

        for i in range(len(self.Ds)):
            result = self.Ds[i](img)
            results.append(result)
            img = self.downsample(img)

        return results


if __name__ == '__main__':
    import torch
    from torchinfo import summary

    class Args:
        def __init__(self, img_size, in_channel=3):
            self.img_size = img_size
            self.d_base_channel = 512
            self.in_channel = in_channel
            self.num_patchD = 2

    img_sizes = [(64, 64), (48, 80), (256, 256)]
    for img_size in img_sizes:
        print(f'image size: {img_size[0]}x{img_size[1]}')
        inp_size = (2, 3, *img_size)
        x = torch.randn(*inp_size)
        args = Args(img_size)
        g_disc = choose_global_netD(args=args)
        summary(g_disc, input_size=inp_size, col_names=["input_size", "output_size", "num_params"])
        #print('output shape of global disc:', g_disc(x).shape)
        if args.num_patchD > 0:
            p_disc = choose_patch_netD(args=args)
            print('output shape of patch disc:', p_disc(x).shape)
        disc = Discriminator(args)
        results = disc(x)
        for i, r in enumerate(results):
            print(f'output shape of disc{i}:', r.shape)
        print('')