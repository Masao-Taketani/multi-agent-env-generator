import os
import shutil
from glob import glob
from tqdm import tqdm
tqdm.monitor_interval = 0
from PIL import Image
import argparse


SIZE = (64, 64)
EDGE = 256
LEFT, TOP, RIGHT, BOTTOM = 40, 22, EDGE-40, EDGE-58

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, default='datasets/tmp/carla/')
parser.add_argument('--out_dir', type=str, default='datasets/carla/')
args = parser.parse_args()


if __name__ == '__main__':
    # Loop over data{1-6} dirs
    for data_dir in glob(os.path.join(args.in_dir, '*')):
        data_dname = os.path.basename(data_dir)
        # Loop over episodes for each data dir
        for epd in tqdm(glob(os.path.join(data_dir, '*'))):
            # only handle epsode dirs. Skip joblog.log.
            if not os.path.isdir(epd): continue
            ep = os.path.basename(epd)
            new_ep_dir = os.path.join(args.out_dir, data_dname, ep) 
            os.makedirs(new_ep_dir, exist_ok=True)
            # Copy log data
            shutil.copy(os.path.join(epd, 'info.json'), new_ep_dir)
            # Loop over images for each epsode dir
            for fpath in glob(os.path.join(epd, '*.png')):
                fname = os.path.basename(fpath)
                img = Image.open(fpath)
                img = img.crop((LEFT, TOP, RIGHT, BOTTOM))
                img = img.resize(SIZE, resample=5)
                img.save(os.path.join(new_ep_dir, fname))