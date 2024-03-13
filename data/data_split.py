from argparse import ArgumentParser
import os
import random
import pickle


parser = ArgumentParser()

parser.add_argument('--datapath', type=str)
parser.add_argument('--train_ratio', default=0.7, type=float)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--test_ratio', default=0.2, type=float)
parser.add_argument('--random_seed', default=1234, type=float)
args = parser.parse_args()


def normalize_ratio(args):
    # If the given ratio does not sum up to 1, this func normalize the ratios.
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if ratio_sum != 1.0:
        args.train_ratio /= ratio_sum
        args.val_ratio /= ratio_sum
        args.test_ratio /= ratio_sum


if __name__ == '__main__':
    random.seed(args.random_seed)
    normalize_ratio(args)
    episodes = os.listdir(args.datapath)
    train_keys, val_keys, test_keys = [], [], []
    for epi in episodes:
        val = random.random()
        if val < args.train_ratio:
            train_keys.append(epi)
        elif val < args.train_ratio + args.val_ratio:
            val_keys.append(epi)
        else:
            test_keys.append(epi)

    dataset = (train_keys, val_keys, test_keys)
    dataset_name = os.path.basename(os.path.dirname(args.datapath)) \
                   if args.datapath[-1] == '/' else os.path.basename(args.datapath)
    if '_' in dataset_name: dataset_name = dataset_name.split('_')[0]
    fname = f'{dataset_name}_data_split.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'{fname} is created!')