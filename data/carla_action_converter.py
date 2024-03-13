import os
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
tqdm.monitor_interval = 0
import json


parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, default='datasets/carla/')
args = parser.parse_args()


def normalize_action(angv, speed):
    # Normalize angular velocity
    # + right  - left
    angv = (angv - (-0.40)) / 20.45
    # Normalize speed
    speed = (speed - 18.2) / 3.62
    return angv, speed

def create_action_log(out_dpath, log):
    action_log = []
    key = 'first_0'
    for action in log['data']:
        angv, speed = normalize_action(action['angular_velocity'][2], action['speed'])
        action = {key: [angv, speed]}
        action_log.append(action)

    np.savez_compressed(os.path.join(out_dpath, 'action_log'),
                        actions=np.array(action_log))
    

if __name__ == '__main__':
    # Loop over data{1-6} dirs
    for data_dir in glob(os.path.join(args.in_dir, '*')):
        data_dname = os.path.basename(data_dir)
        # Loop over episodes for each data dir
        for epd in tqdm(glob(os.path.join(data_dir, '*'))):
            # only handle epsode dirs. Skip joblog.log.
            if not os.path.isdir(epd): continue
            json_path = os.path.join(epd, 'info.json')
            log = json.load(open(json_path, 'rb'))
            create_action_log(epd, log)
            os.remove(json_path)