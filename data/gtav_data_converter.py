from argparse import ArgumentParser
import os
import gzip
import pickle
from PIL import Image
import numpy as np
from tqdm import tqdm


parser = ArgumentParser()

parser.add_argument('--data_path', type=str, help="specify GTAV dataset dir path")
parser.add_argument('--split_ratio', type=float, default=0.9, 
                    help="Train Validation split ratio. Specify the ratio for training \
                          data")
parser.add_argument('--out_path', type=str, default='datasets/gtav', \
                    help="specify output dir path")
args = parser.parse_args()


def get_datalist_and_length(args):
    datalist = os.listdir(args.data_path)
    num_data = len(datalist)
    return datalist, num_data

def create_split_file(args):
    split_idx = int(num_data * args.split_ratio)
    train_keys = []
    val_keys = []
    for epi_idx in range(num_data):
        train_keys.append(str(epi_idx)) if epi_idx < split_idx else val_keys.append(str(epi_idx))

    dataset = (train_keys, val_keys)
    fname = 'gtav_data_split.pkl'
    
    with open(fname, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'{fname} is created!')

def create_episode_dir(dpath):
    os.makedirs(dpath, exist_ok=True)

def create_images(dpath, np_imgs):
    for i in range(np_imgs.shape[0]):
        create_png(dpath, np_imgs[i], i)    

def create_png(dpath, np_img, idx):
    Image.fromarray(np_img).save(os.path.join(dpath, str(idx) + '.png'))

def create_action_log(dpath, actions):
    action_log = []
    key = 'first_0'
    for action in actions:
        action = {key: action}
        action_log.append(action)

    np.savez_compressed(os.path.join(dpath, 'action_log'),
                        actions=np.array(action_log))


if __name__ == '__main__':
    files, num_data = get_datalist_and_length(args)
    create_split_file(args)

    for ep_idx in tqdm(range(num_data)):
        dpath = os.path.join(args.out_path, str(ep_idx))
        create_episode_dir(dpath)

        gz_path = os.path.join(args.data_path, files[ep_idx])
        with gzip.open(gz_path, 'rb') as f:
            data = pickle.load(f)

        create_images(dpath, data['observations'])
        create_action_log(dpath, data['actions'])