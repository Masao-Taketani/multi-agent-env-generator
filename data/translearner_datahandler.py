import torch
from torch.utils import data
import pickle
import os
import random
import numpy as np

from utils import check_arg, list_to_dict


def get_custom_dataset(args, train, force_noshuffle=False, getLoader=True):

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    dataset = []

    shuffle = True if train else False
    #shuffle = True if args.play else shuffle

    if force_noshuffle:
        shuffle = False

    dataset.append(GenericDataset(args, train=train, datadir=args.data_dir))

    iter_len = 0
    if getLoader:
        dloader = []
        batch_size = args.batch_size if train else args.val_batch_size
        for dset in dataset:
            iter_len += len(dset)
            dloader.append(data.DataLoader(dset, batch_size=batch_size,
                    num_workers=args.num_workers, pin_memory=True, shuffle=shuffle, drop_last=True, collate_fn=collate_fn))
        if len(dataset) == 1 and not args.test:
            return dloader[0], iter_len
        return dloader, iter_len
    else:
        return dataset, len(dataset)


class GenericDataset(data.Dataset):

    def __init__(self, args, train, datadir=''):
        self.args = args
        self.train = train
        self.samples = []
        self.layout_memory = check_arg(self.args, 'layout_memory')
        self.continuous_action = check_arg(self.args, 'continuous_action')
        self.predict_logvar = check_arg(self.args, 'predict_logvar')
        self.learn_interpolation = check_arg(self.args, 'learn_interpolation')
        self.no_duplicate = check_arg(self.args, 'no_duplicate')

        paths = []
        
        if 'carla' in args.dataset:
            try:
                train_keys, val_keys, tst_keys = pickle.load(open('carla_data_split.pkl', 'rb'))
            except:
                train_keys, val_keys, tst_keys = pickle.load(open('../carla_data_split.pkl', 'rb'))

            train_keys = list_to_dict(train_keys)
            val_keys = list_to_dict(val_keys)
            tst_keys = list_to_dict(tst_keys)

            root_dir = datadir
            for fname in os.listdir(root_dir):
                cur_file = os.path.join(datadir, fname)
                if not '.npy' in fname:
                    continue

                key = fname.split('.')[0]
                key = key.replace('_', '/')
                do = False
                if (train and key in train_keys) or (not train and key in val_keys) or (args.test and key in tst_keys):
                    do = True
                if not do:
                    continue
                paths.append([key, cur_file])

        elif 'pong' in self.args.dataset or 'boxing' in self.args.dataset or 'gtav' in self.args.dataset:
            train_keys, val_keys, _ = pickle.load(open(f'{self.args.dataset}_data_split.pkl', 'rb'))
            train_keys = list_to_dict(train_keys)
            val_keys = list_to_dict(val_keys)
            
            root_dir = datadir
            for fname in os.listdir(root_dir):
                if not '.npy' in fname:
                    continue
                cur_file = os.path.join(datadir, fname)
                
                key = fname.split('.')[0]
                do = False
                if (train and key in train_keys) or (not train and key in val_keys):
                    do = True
                if not do:
                    continue
                paths.append([key, cur_file])
        else:
            assert 0, f'dataset {self.args.dataset} is not supported.'

        random.Random(4).shuffle(paths)
        if check_arg(self.args, 'num_chunk') and self.args.num_chunk > 0:
            num_chunk = self.args.num_chunk
            cur_ind = self.args.cur_ind
            chunk_size = len(paths) // num_chunk
            if cur_ind == num_chunk-1:
                paths = paths[cur_ind*chunk_size:]
            else:
                paths = paths[cur_ind*chunk_size:(cur_ind+1)*chunk_size]

        self.samples = paths
        data_type = 'train' if train else 'valid'
        print(f'\n\n----{data_type} num Episodes: {len(paths)} \n\n')
        if not train:
            self.args.val_batch_size = len(paths)

    def parse_action_ma(self, cur_a):
        actions = []
        dtype = 'float32' if 'carla' in self.args.dataset or self.continuous_action else 'int64'
        for key in cur_a.keys():
            actions.append(np.asarray([cur_a[key]]).astype(dtype))
        return actions
    
    def get_neg_action_ma(self, a_ts):
        neg_a_ts = []
        if 'carla' in self.args.dataset or self.continuous_action:
            for i in range(len(a_ts)):
                neg_a_ts.append(np.asarray([a_ts[i]]).astype('float32'))
        else:
            for i in range(len(a_ts)):
                neg_action = random.randint(0, self.args.action_space - 1)
                while neg_action == a_ts[i][0]:
                    neg_action = random.randint(0, self.args.action_space - 1)
                neg_a_ts.append(np.asarray([neg_action]).astype('int64'))
        return neg_a_ts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fn = self.samples[idx]
        try:
            data = np.load(fn[1], allow_pickle=True).item()
            len_episode = len(data['actions'])
        except:
            print('dataloader error: ')
            print(fn)
            return None

        if 'pong' in self.args.dataset or 'boxing' in self.args.dataset \
             or 'gtav' in self.args.dataset or 'carla' in self.args.dataset:
            if self.args.use_neg_actions == True:
                states = []
                actions = []
                neg_actions = []
                for i in range(self.args.num_agents):
                    actions.append([])
                    neg_actions.append([])
            else:  
                states = []
                actions = []
                for i in range(self.args.num_agents):
                    actions.append([])

            frame_key, act_key = 'latent_imgs', 'actions'

            # -1 indicates that vis length needs to have max_seq_len + 1 for input and label
            ep_len = len_episode - self.args.max_seq_len - 1
            if self.args.test:
                start_pt = 0  ## start from the first screen for testing
            else:
                start_pt = random.randint(0, ep_len)

            i = 0
            while i < self.args.max_seq_len:
                if start_pt + i >= len(data[frame_key]):
                    cur_s = data[frame_key][len(data[frame_key]) - 1]
                    cur_a = data[act_key][len(data[frame_key]) - 1]
                else:
                    cur_s = data[frame_key][start_pt + i]
                    cur_a = data[act_key][start_pt + i]

                # expand the first dimension for frames so that it can be concatenated
                # w.r.t seq axis
                s_t = cur_s[np.newaxis, :]
                a_ts = self.parse_action_ma(cur_a)

                # save
                states.append(s_t)
                for j in range(self.args.num_agents):
                    actions[j].append(a_ts[j])

                if self.args.use_neg_actions == True:
                    if 'carla' in self.args.dataset or self.continuous_action:
                        neg_carla_actions = []
                        for key in cur_a.keys():
                            # sample negative action within the episode
                            rand_ind = random.randint(start_pt, start_pt+self.args.max_seq_len - 1)
                            while rand_ind == start_pt + i:
                                rand_ind = random.randint(start_pt, start_pt+self.args.max_seq_len - 1)
                            neg_carla_actions.append(data[act_key][rand_ind][key])
                        neg_a_ts = self.get_neg_action_ma(neg_carla_actions)
                    else:
                        neg_a_ts = self.get_neg_action_ma(a_ts)
                    for j in range(self.args.num_agents):
                        neg_actions[j].append(neg_a_ts[j])
                i = i + 1
            
            # this is for a teacher for the last time step for vis
            states.append(data[frame_key][start_pt + i][np.newaxis, :])

            del data
            if self.args.use_neg_actions == True:
                return states, actions, neg_actions
            else:    
                return states, actions