import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
import cv2
import pickle
from utils import check_arg, list_to_dict


class ImageDataset(Dataset):
    def __init__(self, path, img_size, dataset, args=None, train=True):
        datadir = path
        paths = []
        self.img_size = img_size
        self.dataset = dataset
        self.args = args
        self.width_mul = 1
        if check_arg(args, 'width_mul'):
            self.width_mul = args.width_mul
        self.crop_input = 0
        if check_arg(args, 'crop_input'):
            self.crop_input = args.crop_input

        self.img_transform = None
        self.is_pilotnet = False
        if self.dataset == 'gibson':
            pass # data not released due to legal issue
        elif 'carla' in self.dataset:
            episode_end = 80
            self.episode_end = episode_end
            train_keys, val_keys, tst_keys = pickle.load(open('carla_data_split.pkl', 'rb'))
            train_keys = list_to_dict(train_keys)
            val_keys = list_to_dict(val_keys)

            paths = []
            root_dirs = datadir.split(',')
            for root_dir in root_dirs:
                if not os.path.isdir(root_dir):
                    continue
                for episode in os.listdir(root_dir):
                    if not os.path.isdir(os.path.join(root_dir, episode)):
                        continue
                    key = os.path.basename(root_dir) +'/'+ episode
                    do = False

                    if (train and key in train_keys) or (not train and key in val_keys): # or not args.distributed:
                        ## if not distributed, just local debugging
                        do = True
                    if not do:
                        continue

                    for i in range(0, episode_end):
                        fname1 = str(i) + '.png'
                        fpath1 = os.path.join(root_dir, episode, fname1)
                        paths.append(fpath1)

        elif 'pilotnet' in self.dataset:
            pass # data not released due to legal issue
        elif 'pong' in self.dataset or 'boxing' in self.dataset or 'gtav' in self.dataset:
            train_keys, val_keys, _ = pickle.load(open(f'{self.dataset}_data_split.pkl', 'rb'))
            train_keys = list_to_dict(train_keys)
            val_keys = list_to_dict(val_keys)
            paths = []
            root_dir = datadir

            for episode in os.listdir(root_dir):
                key = episode

                if (train and key in train_keys) or (not train and key in val_keys): # or not args.distributed:
                    # need to sbtract 1 since each episode dir includes an action log
                    episode_end = len(os.listdir(os.path.join(root_dir, episode))) - 1
                    for i in range(0, episode_end):
                        fname1 = str(i) + '.png'
                        fpath1 = os.path.join(root_dir, episode, fname1)
                        paths.append(fpath1)

        else:
            print('dataset not supported')
            exit(-1)
        print('\n\n\n\n\n\n' + self.dataset + ', Number of data: ' + str(len(paths)) + '\n\n\n\n\n\n')
        random.Random(4).shuffle(paths)

        #if not train:
        #    # temporary: don't spend too much time on validation for faster training
        #    #if not args.distributed:
        #    paths = paths[:min(len(paths), 100)]
        
        self.samples = paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        try:
            fn = self.samples[index]
            #if 'carla' in self.dataset:
            #    cur_s = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
            #    cur_s = cv2.cvtColor(cur_s, cv2.COLOR_BGR2RGB)
            #elif 'pong' in self.dataset:
            cur_s = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
            cur_s = cv2.cvtColor(cur_s, cv2.COLOR_BGR2RGB)

            cur_s = cur_s / 255.0

            if self.crop_input and self.is_pilotnet:
                #### pilotnet only to match bdd for now
                h_offset = int(150/512 * cur_s.shape[0])
                h_size = int(300/512 * cur_s.shape[0])
                w_size = int(533/814 * cur_s.shape[1])
                w_offset = (cur_s.shape[1] - w_size) // 2
                cur_s = cur_s[h_offset:h_offset+h_size, w_offset:w_offset+w_size]

            if cur_s.shape[1] != self.img_size[0] and cur_s.shape[2] != int(self.img_size[1]*self.width_mul):
                cur_s = cv2.resize(cur_s, (int(self.img_size[1]*self.width_mul),  self.img_size[0]))
            s_t = (np.transpose(cur_s, axes=(2, 0, 1))).astype('float32')
            s_t = torch.FloatTensor((s_t - 0.5) / 0.5)
            if self.img_transform is not None:
                s_t = self.img_transform(s_t)
        except:
            print(f'Error occurred for a file {fn} when __getitem__ is applied.')
            return None

        return s_t


class EpisodeDataset(Dataset):
    def __init__(self, path, img_size, dataset, args=None):
        datadir = path
        paths = []
        self.img_size = img_size
        self.dataset = dataset
        self.args = args
        self.width_mul = 1
        if check_arg(args, 'width_mul'):
            self.width_mul = args.width_mul

        self.crop_input = 0
        if check_arg(args, 'crop_input'):
            self.crop_input = args.crop_input

        root_dirs = datadir.split(',')
        if 'gibson' in self.dataset:
            pass # data not released
        elif 'carla' in self.dataset:
            episode_end = 80
            self.episode_end = episode_end
            train_keys, val_keys, tst_keys = pickle.load(open('carla_data_split.pkl', 'rb'))
            train_keys = list_to_dict(train_keys)
            val_keys = list_to_dict(val_keys)
            tst_keys = list_to_dict(tst_keys)

            paths = []
            root_dirs = datadir.split(',')
            for root_dir in root_dirs:
                if not os.path.isdir(root_dir):
                    continue

                for episode in os.listdir(root_dir):
                    if os.path.isdir(os.path.join(root_dir, episode)):
                        key = os.path.basename(root_dir) + '/' + episode
                        do = False
                        if (key in train_keys) or (key in val_keys) or (args.test and key in tst_keys):
                            do = True
                        if not do:
                            continue
                        cur_paths = []
                        for i in range(80):
                            cur_paths.append(os.path.join(root_dir, episode, str(i)+'.png'))
                        key = key.replace('/', '_')
                        paths.append([key, cur_paths])
        elif 'pong' in self.dataset or 'boxing' in self.dataset or 'gtav' in self.dataset:
            train_keys, val_keys, _ = pickle.load(open(f'{self.dataset}_data_split.pkl', 'rb')) 
            train_keys = list_to_dict(train_keys)
            val_keys = list_to_dict(val_keys)
            paths = []
            root_dir = datadir

            for episode in os.listdir(root_dir):
                key = episode
                do = False

                if (key in train_keys) or (key in val_keys):
                    do = True
                if not do:
                    continue
                cur_paths = []
                # need to sbtract 1 since each episode dir includes an action log
                episode_end = len(os.listdir(os.path.join(root_dir, episode))) - 1
                for i in range(episode_end):
                    cur_paths.append(os.path.join(root_dir, episode, str(i)+'.png'))
                paths.append([key, cur_paths])

        elif 'pilotnet' in self.dataset:
            pass # data not released
        else:
            print('dataset not supported')
            exit(-1)

        if args.num_chunk > 0:
            num_paths = len(paths)
            chunk_size = num_paths // args.num_chunk
            start = chunk_size * args.cur_ind
            end = chunk_size * (args.cur_ind + 1) if args.cur_ind < args.num_chunk - 1 else num_paths
            paths = paths[start:end]

        print('\n\n\n\n\n\n' + self.dataset + ', Number of episodes: ' + str(len(paths)) + '\n\n\n\n\n\n')
        self.samples = paths

    def __len__(self):
        return len(self.samples)

    def load_and_process_image(self, img_path):
        cur_s = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        cur_s = cv2.cvtColor(cur_s, cv2.COLOR_BGR2RGB)

        cur_s = cur_s / 255.0
        if self.crop_input:
            #### pilotnet only to match bdd for now
            h_offset = int(150/512 * cur_s.shape[0])
            h_size = int(300/512 * cur_s.shape[0])
            w_size = int(533/814 * cur_s.shape[1])
            w_offset = (cur_s.shape[1] - w_size) // 2
            cur_s = cur_s[h_offset:h_offset+h_size, w_offset:w_offset+w_size]

        if cur_s.shape[1] != self.img_size[0] and cur_s.shape[2] != int(self.img_size[1]*self.width_mul):
            cur_s = cv2.resize(cur_s, (int(self.img_size[1]*self.width_mul),  self.img_size[0]))
        s_t = (np.transpose(cur_s, axes=(2, 0, 1))).astype('float32')
        s_t = (s_t - 0.5) / 0.5
        return s_t

    def __getitem__(self, index):
        try:
            key, episode_path = self.samples[index]
            imgs = []
            for fname in episode_path:
                img_path = fname
                cur_s = self.load_and_process_image(img_path)
                imgs.append(cur_s)
        except:
            print(f'Error occurred for a file {fname} when __getitem__ is applied.')
            return None

        return np.array(imgs).astype('float32'), episode_path, key