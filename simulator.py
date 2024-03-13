import torch
import argparse
import keyboard
import cv2
import time
import numpy as np
import random
from PIL import Image

from models.enc_dec import EncoderDecoder
from models.transformer import TransitionLearner
from models.transformer_old import TransitionLearner as TransitionLearnerOld
from models.transformer_xl import TransformerXL


def show_and_return_init_img(init_img_path):
        if init_img_path is None:
            # Load the dataset so we can get some initial image
            ##!! Replace with some examle set-aside images
            train_loader = dataloader.get_custom_dataset(opts, set_type=0, getLoader=True)
            data_iters, train_len = [], 99999999999
            data_iters.append(iter(train_loader))
            if len(data_iters[-1]) < train_len:
                train_len = len(data_iters[-1])
            states, actions, _ = utils.get_data(data_iters, opts)
        else:
            # Load starting image
            img = cv2.imread(init_img_path)
        cv2.imshow(f'inference', img)
        cv2.waitKey(1000)
        return convert_from_numpy_to_torch(img)     

def convert_from_numpy_to_torch(img):
    img = img[...,::-1]
    img = (np.transpose(img, axes=(2, 0, 1)) / 255.).astype('float32')
    img = (img - 0.5) / 0.5
    return torch.unsqueeze(torch.from_numpy(img), 0)

def render(img):
    img = convert_from_torch_to_numpy(img)[...,::-1]
    cv2.imshow(f'inference', img)

def convert_from_torch_to_numpy(img):
    img = torch.squeeze(img, 0)
    img = img.cpu().numpy()
    img = np.transpose(img, axes=(1, 2, 0))
    return ((img+1)*127.5).astype(np.uint8)

def select_action_for_pong(fire, right, left, device):
    if keyboard.is_pressed(fire):
        # Fire
        action = torch.tensor([1], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(right):
        # Move right
        action = torch.tensor([2], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(left):
        # Move left
        action = torch.tensor([3], dtype=torch.int64).to(device)
    else:
        # No operation
        action = torch.tensor([0], dtype=torch.int64).to(device)
    
    return action

def select_action_for_boxing(fire, up, right, left, down, device):
    if keyboard.is_pressed(fire):
        # Fire
        action = torch.tensor([1], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(up):
        # Move up
        action = torch.tensor([2], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(right):
        # Move right
        action = torch.tensor([3], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(left):
        # Fire left
        action = torch.tensor([4], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(down):
        # Fire down
        action = torch.tensor([5], dtype=torch.int64).to(device)
    else:
        # No operation
        action = torch.tensor([0], dtype=torch.int64).to(device)
    
    return action

def select_action_for_gtav(left, right, device):
    if keyboard.is_pressed(left):
        # Move left
        action = torch.tensor([0], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(right):
        # Move right
        action = torch.tensor([2], dtype=torch.int64).to(device)
    else:
        # No operation
        action = torch.tensor([1], dtype=torch.int64).to(device)
    return action

def request_and_return_agent_actions_pong(num_agents, device):
    if num_agents == 2:
        # action for player1
        a1 = select_action_for_pong('w', 'd', 'a', device)
        # action for player2
        a2 = select_action_for_pong('i', 'l', 'j', device)
        return a1, a2
    elif num_agents == 4:
        # action for player1
        a1 = select_action_for_pong('w', 'd', 'a', device)
        # action for player2
        a2 = select_action_for_pong('t', 'h', 'f', device)
        # action for player3
        a3 = select_action_for_pong('i', 'l', 'j', device)
        # action for player4
        a4 = select_action_for_pong('s', 'c', 'z', device)
        return a1, a2, a3, a4
    else:
        assert 0, 'num_agents for pong has to be either 2 or 4' 

def request_and_return_agent_actions_boxing(device):
    # action for player1
    a1 = select_action_for_boxing('e', 'w', 'd', 'a', 's', device)
    
    # action for player2
    a2 = select_action_for_boxing('u', 'i', 'l', 'j', 'k', device)
        
    return a1, a2

def request_and_return_agent_actions_gtav(device):
    a = select_action_for_gtav('a', 'd', device)
    return (a,)

def create_action_log(args, mul_actions):
    assert 0 < args.num_agents < 5, 'create_action_log func accepts num agents up to 4.'
    import os
    action_log = []
    keys = ['first_0', 'second_0', 'third_0', 'fourth_0'][:args.num_agents]
    for mul_action in mul_actions:
        act_dic = {}
        for key, action in zip(keys, mul_action):
            act_dic[key] = action.item()
        action_log.append(act_dic)

    os.makedirs(args.action_log_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.action_log_dir, 'action_log'),
                        actions=np.array(action_log))
    print(f"Action log is created at '{args.action_log_dir}'.")

def run_simulator(args, encdec, translearner, device, run_steps=None):

    cur_img = show_and_return_init_img(args.init_img_path).to(device)
    i = 0

    if args.rec:
        if args.cv2_rec:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            rec_path = args.rec_path_wo_ext + '.mp4'
            v = cv2.VideoWriter(rec_path, fourcc, 10.0, (args.img_size[1], args.img_size[0]))
            np_img = convert_from_torch_to_numpy(cur_img)[...,::-1]
            v.write(np_img)
        elif args.gif_rec:
            np_imgs = []
            np_imgs.append(convert_from_torch_to_numpy(cur_img))

    if args.make_action_log:
        if not args.action_log_dir:
            assert 0, 'When make_action_log flag is used, you need to specify \
                       a directory path to save the action log.'
        mul_actions = []

    while True:
        frame_start_time = time.time()

        if keyboard.is_pressed('q'):
            break

        if args.random_action:
            actions = []
            for _ in range(args.num_agents):
                actions.append(torch.tensor([random.randint(0, args.action_space-1)], dtype=torch.int64).to(device))
        else:
            if args.dataset == 'pong':
                actions = request_and_return_agent_actions_pong(args.num_agents, device)
            elif args.dataset == 'boxing':
                actions = request_and_return_agent_actions_boxing(device)
            elif args.dataset == 'gtav':
                actions = request_and_return_agent_actions_gtav(device)
        
        if args.make_action_log:
            mul_actions.append(list(actions))

        cur_img_emb = encdec.enc(cur_img)
        if args.old:
            fut_img_emb = translearner.test_step(cur_img_emb, *actions)
        else:
            fut_img_emb = translearner.test_step(cur_img_emb, actions)
        cur_img = encdec.dec(fut_img_emb)
        render(cur_img)

        if args.rec:
            if args.cv2_rec:
                np_img = convert_from_torch_to_numpy(cur_img)[...,::-1]
                v.write(np_img)
            elif args.gif_rec:
                np_imgs.append(convert_from_torch_to_numpy(cur_img))
        cv2.waitKey(1)

        wait = 1/args.fps - (time.time() - frame_start_time)
        if run_steps is not None and i == run_steps:
            break
        i += 1
        if wait > 0:
            time.sleep(wait)

    if args.make_action_log:
        create_action_log(args, mul_actions)
    
    if args.rec and args.gif_rec:
        pil_imgs = []
        for img in np_imgs:
            pil_imgs.append(Image.fromarray(img))
        rec_path = args.rec_path_wo_ext + '.gif'
        pil_imgs[0].save(rec_path, save_all=True, append_images=pil_imgs[1:], 
                optimize=False, duration=20, loop=0)
        

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--encdec_ckpt', type=str, default=None)
    parser.add_argument('--trans_ckpt', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='pong')
    parser.add_argument('--init_img_path', type=str, default='')
    parser.add_argument('--width_mul', type=float, default=1)
    parser.add_argument('--crop_input', type=int, default=0)
    parser.add_argument('--rec', action='store_true')
    parser.add_argument('--cv2_rec', action='store_true')
    parser.add_argument('--gif_rec', action='store_true')
    parser.add_argument('--img_size', type=str, default='64x64', help='heightxwidth')
    parser.add_argument('--rec_path_wo_ext', type=str, default='test')
    parser.add_argument('--random_action', action='store_true')
    parser.add_argument('--run_steps', type=int, default=-1)
    parser.add_argument('--attn_mask_type', type=str, default='')

    parser.add_argument('--trans_type', type=str)
    # to overwrite mem_len used during training
    parser.add_argument('--mem_len', type=int, default=-1)
    parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens.\
                          only used for trans xl models')
    parser.add_argument('--fps', type=int, default=30)

    parser.add_argument('--action_space', type=int, default=-1,
                        help='old model args does not include this argment. \
                              If that happens, you need to specify it by yourself.')
    parser.add_argument('--make_action_log', action='store_true')
    parser.add_argument('--action_log_dir', type=str, default='', 
                        help='When make_action_log flag is used, you need to specify \
                              a directory path to save the action log.')

    args = parser.parse_args()

    # load ckpts and init models
    print('load encdec model:', args.encdec_ckpt)
    encdec_saved_file = torch.load(args.encdec_ckpt)
    encdec_arg = encdec_saved_file['args']
    encdec = EncoderDecoder(encdec_arg).to(device)
    encdec.load_state_dict(encdec_saved_file['encdec'], strict=True)
    del encdec_saved_file

    print('load translearner model:', args.trans_ckpt)
    trans_saved_file = torch.load(args.trans_ckpt)
    if 'args' in trans_saved_file:
        trans_args = trans_saved_file['args']
    elif 'opts' in trans_saved_file:
        trans_args = trans_saved_file['opts']
    else:
        raise KeyError('Neither `args` nor `opts` exists. Need to check your saved format!')
    
    trans_args.continuous_action = getattr(trans_args, 'continuous_action', False)

    #print('bs:', vars(trans_arg)['batch_size'])
    if not 'num_action_history' in vars(trans_args).keys():
        trans_args.num_action_history = trans_args.max_seq_len

    trans_type = None
    if getattr(trans_args, "trans_type", None) is not None:
        trans_type = getattr(trans_args, "trans_type")
        if trans_type == 'vanila': trans_type = 'vanilla'
    else:
        if getattr(args, "trans_type") is None:
            raise KeyError(f"You need to specify 'trans_type'.")
        elif args.trans_type == 'vanilla':
            trans_type = 'vanilla'
        elif args.trans_type == 'xl':
            trans_type = 'xl'
        else:
            raise NameError(f'trans_type {args.trans_type} does not exist!')
    
    if trans_type == 'vanilla':
        if args.attn_mask_type != '':
            trans_args.attn_mask_type = args.attn_mask_type
        elif getattr(trans_args, 'attn_mask_type', None) is None:
            raise KeyError('Need to specify proper attn_mask_type.')
        
        num_agents = getattr(trans_args, 'num_agents', None)
        if num_agents is None:
            translearner = TransitionLearnerOld(trans_args).to(device)
            args.num_agents = 2
            args.old = True
        else:
            translearner = TransitionLearner(trans_args).to(device)
            args.num_agents = num_agents
            args.old = False

        translearner.load_state_dict(trans_saved_file['translearner'], strict=True)
        del trans_saved_file
        batch_size = 1
        translearner.init_queries(device, batch_size)
    elif trans_type == 'xl':
        trans_args.same_length = args.same_length
        if args.mem_len > -1: trans_args.mem_len = args.mem_len
        translearner = TransformerXL(trans_args.n_layer, trans_args.n_head, trans_args.d_model, 
                                     trans_args.d_head, trans_args.d_inner, trans_args.dropout, 
                                     trans_args.dropatt, trans_args.action_space, trans_args.d_embed, 
                                     trans_args.pre_lnorm, trans_args.max_seq_len, trans_args.ext_len, 
                                     trans_args.mem_len, trans_args.same_length, 
                                     trans_args.clamp_len).to(device)
        translearner.load_state_dict(trans_saved_file['translearner'], strict=True)
        del trans_saved_file
        translearner.init_queries_and_mems(device)

    encdec.eval()
    translearner.eval()
    
    args.dataset = trans_args.dataset
    args.img_size = tuple([int(i) for i in args.img_size.split('x')])
    run_steps = args.run_steps if args.run_steps > 0 else None

    if args.random_action: args.action_space = trans_args.action_space
    with torch.no_grad():
        if args.rec:
            assert args.cv2_rec == True or args.gif_rec == True, \
                   'Either cv2_rec or gif_rec has to be True to record'
            
        run_simulator(args, encdec, translearner, device, run_steps)