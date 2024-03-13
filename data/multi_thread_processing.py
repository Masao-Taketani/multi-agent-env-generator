"""
For this multi-threading code, we referred to 
https://github.com/ctallec/world-models/blob/master/data/generation_script.py
"""

from os import makedirs
from os.path import join
import argparse
from multiprocessing import Pool
from subprocess import call
from glob import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='boxing', type=str, help="Select dataset among boxing, pong, or \
                                                                   quadrapong")
parser.add_argument('--num_eps', type=int, help="Total number of episodes.")
parser.add_argument('--num_threads', type=int, help="Number of threads")
parser.add_argument('--data_dir', type=str, help="Directory to store rollout "
                    "directories of each thread")
parser.add_argument('--img_size', type=str, default='64x64', help="Resize each image to the specified size. \
                                                                   if None, use original image size")
parser.add_argument('--num_agents', default=2, type=int, help="Only used for pong environment")
args = parser.parse_args()
num_eps_per_thread = args.num_eps // args.num_threads + 1
if args.num_agents == 2 or args.dataset == 'quadrapong':
    ROOT_DIR = join(args.data_dir, args.dataset)
else:
    ROOT_DIR = join(args.data_dir, args.dataset + f'_num_agts_{args.num_agents}')


def _threaded_generation(i):
    assert args.dataset == 'boxing' or args.dataset == 'pong' or args.dataset == 'quadrapong', \
                           f"It does not support a dataset name '{args.dataset}'"
    makedirs(ROOT_DIR, exist_ok=True)
    path_prefix = join(ROOT_DIR, 'thread_{}'.format(i))
    cmd = ['xvfb-run', '-a', '-s', '"-screen 0 1400x900x24"']
    cmd += ['--server-num={}'.format(i + 1)]
    cmd += ["python3", "data/data_creation.py", "--dataset", f'{args.dataset}', 
            "--path_prefix", f"{path_prefix}", "--num_eps", f"{num_eps_per_thread}",
            "--num_agents", f"{args.num_agents}"]
    if args.img_size is not None:
        cmd += ["--img_size", f'{args.img_size}']
    cmd = " ".join(cmd)
    print(cmd)
    call(cmd, shell=True)
    return True


if __name__ == '__main__':
    with Pool(args.num_threads) as p:
        p.map(_threaded_generation, range(args.num_threads))
    # change file names
    thread_ep_dirs = glob(join(ROOT_DIR, 'thread_*'))
    for i, thread_dir in enumerate(thread_ep_dirs):
        new_dir_name = join(ROOT_DIR, str(i))
        os.makedirs(new_dir_name, exist_ok=True)
        os.rename(thread_dir, new_dir_name)
