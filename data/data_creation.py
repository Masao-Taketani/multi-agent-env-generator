from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import cv2
import random
from PIL import Image
import os


parser = ArgumentParser()

parser.add_argument('--dataset', default='boxing', type=str, help="select dataset among boxing, pong")
parser.add_argument('--num_eps', default=1000, type=int, help="total number of episodes to create the dataset")
parser.add_argument('--path_prefix', type=str, help="select dataset among boxing, pong")
parser.add_argument('--img_size', type=str, default=None, help="Resize each image to the specified size. \
                                                                if None, use original image size")
parser.add_argument('--num_agents', default=2, type=int, help="Only used for pong environment")

args = parser.parse_args()


def get_edges(img_size, edges_to_crop):
    h, w , _ = img_size
    top, bottom, left, right = edges_to_crop
    return (top, h - bottom, left, w - right)


def crop_image(img, edges):
    top, bottom, left, right = edges
    return img[top:bottom, left:right]


def run_env(parallel_env, num_ep, path_prefix, new_size, edges_to_crop):
    observations = parallel_env.reset()
    action_log = []
    key = parallel_env.agents[0]
    img_size = observations[key].shape
    edges = get_edges(img_size, edges_to_crop)
    step = 0
    
    # used for boxing dataset
    init_step = 37
    move_diag = 19
    fname_step = 0

    dirname = path_prefix + '_' + str(num_ep)
    os.makedirs(dirname, exist_ok=True)
    
    while True:
        img = observations[key]
        if edges_to_crop:
            img = crop_image(img, edges)

        if new_size is not None:
            img = cv2.resize(img, dsize=new_size, interpolation=cv2.INTER_CUBIC)

        if args.dataset == 'boxing':
            if step > init_step:
                actions = {agent: random.randint(0, 5) for agent in parallel_env.agents}
                Image.fromarray(img).save(os.path.join(dirname, str(fname_step) + '.png'))
                action_log.append(actions)
                fname_step += 1
            else:
                if step > move_diag:
                    # white: move down, black: move up
                    actions = {'first_0': 5, 'second_0': 2}
                else:
                    # white: move downright, black: move upleft
                    actions = {'first_0': 8, 'second_0': 7}
        elif args.dataset == 'pong':
            # full action choises
            #actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
            # restricted action choises
            actions = {agent: random.randint(0, 3) for agent in parallel_env.agents}
            # set an init step as 57 since the players do not start playing with a ball till then.
            if step > 56:
                Image.fromarray(img).save(os.path.join(dirname, str(fname_step) + '.png'))
                action_log.append(actions)
                fname_step += 1

        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
        if terminations[key]:
            break
        step += 1

    #np.savez_compressed(path_prefix + '_ep_{}'.format(num_ep),
    #                    states=np.array(state_log),
    #                    actions=np.array(action_log))
    np.savez_compressed(os.path.join(dirname, 'action_log'),
                        actions=np.array(action_log))


if __name__ == '__main__':
    if args.dataset == 'boxing':
        from pettingzoo.atari import boxing_v2
        parallel_env = boxing_v2.parallel_env()
        top, bottom, left, right = 25, 25, 20, 20
        edges_to_crop = (top, bottom, left, right)
    elif args.dataset == 'pong':
        from pettingzoo.atari import pong_v3
        parallel_env = pong_v3.parallel_env(num_players=args.num_agents)
        top, bottom, left, right = 33, 16, 0, 0
        edges_to_crop = (top, bottom, left, right)

    if args.img_size is None:
        new_size = None
    else:
        new_size = tuple(int(i) for i in args.img_size.split('x'))
    for i in tqdm(range(args.num_eps)):
        run_env(parallel_env, i, args.path_prefix, new_size, edges_to_crop)
