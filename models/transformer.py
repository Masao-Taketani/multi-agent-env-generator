import torch
from torch import nn
import math


class TransitionLearner(nn.Module):
    def __init__(self, args):
        '''
        transformer agent
        '''
        super(TransitionLearner, self).__init__()
        self.args = args
        # encoder and visual embeddings
        self.encoder_vl = EncoderVL(args)

        # embeddings for actions
        if self.args.continuous_action:
            self.emb_act_mlist = nn.ModuleList([nn.Linear(args.action_space, args.hidden_dim) \
                                                for _ in range(args.num_agents)])
        else:
            self.emb_act_mlist = nn.ModuleList([nn.Embedding(args.action_space, args.hidden_dim) \
                                                for _ in range(args.num_agents)])
        # dropouts
        if args.act_drop_ratio > 0.0:
            # nn.Dropout2d drops values channel-wise
            self.dropout_action = nn.Dropout2d(args.act_drop_ratio)

        # final touch
        self.init_weights()
        if args.test:
            self.reset()

    def forward(self, emb_frames, actions):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        emb_actions = self.embed_actions(actions)
        assert emb_frames.shape == emb_actions[0].shape

        # concatenate frames and actions and add encodings
        encoder_out = self.encoder_vl(emb_frames, emb_actions)
        return encoder_out

    def embed_actions(self, actions):
        '''
        embed previous actions
        ''' 
        emb_actions = [ self.emb_act_mlist[i](actions[i]) for i in range(self.args.num_agents)]
        if self.args.act_drop_ratio > 0.0:
            emb_actions = [ self.dropout_action(emb_actions[i]) for i in range(self.args.num_agents)]
        return emb_actions

    def init_queries(self, device, batch_size):
        '''
        reset internal queries (used for real-time execution during test step)
        '''
        self.frame_query = torch.zeros(batch_size, self.args.max_seq_len, self.args.hidden_dim).to(device)
        self.actions_query = []
        for _ in range(self.args.num_agents):
            self.actions_query.append(torch.zeros(batch_size, self.args.max_seq_len, self.args.hidden_dim).to(device))

        self.test_seq_id = 0

    def init_weights(self, init_range=0.1):
        '''
        init embeddings uniformly
        '''
        for i in range(self.args.num_agents):
            self.emb_act_mlist[i].weight.data.uniform_(-init_range, init_range)

    def update_query(self, query, entry):
        if self.args.max_seq_len == self.test_seq_id:
            query[:, :self.args.max_seq_len - 1, :] = query[:, 1:self.args.max_seq_len, :].clone()
            query[:, self.args.max_seq_len - 1, :] = entry
        else:
            query[:, self.test_seq_id, :] = entry
        return query

    def test_step(self, emb_frame, actions):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        emb_actions = self.embed_actions(actions)
        assert emb_frame.shape == emb_actions[0].shape

        self.frame_query = self.update_query(self.frame_query, emb_frame)
        cloned_action_queries = []
        for i in range(len(self.actions_query)):
            self.actions_query[i] = self.update_query(self.actions_query[i], emb_actions[i])
            cloned_action_queries.append(self.actions_query[i].clone())
        # concatenate frames and actions and add encodings
        # As for pos encoding, since every time the below function adds it to emb vecs, 
        # we need to pass original emb vecs every time.
        encoder_out = self.encoder_vl.test_step(self.frame_query.clone(), cloned_action_queries, self.test_seq_id)

        # update self.test_seq_id
        if self.test_seq_id < self.args.max_seq_len:
            self.test_seq_id += 1

        return encoder_out


class EncoderVL(nn.Module):
    def __init__(self, args):
        '''
        transformer encoder for language, frames and action inputs
        '''
        super(EncoderVL, self).__init__()
        self.args = args
        # transofmer layers
        encoder_layer = nn.TransformerEncoderLayer(
            args.hidden_dim, args.transenc_heads, args.ff_dim,
            args.tranenclayer_drop_ratio)
        self.enc_transformer = nn.TransformerEncoder(
            encoder_layer, args.num_transenclayer)

        # position encodings
        if args.pos_enc_type == 'original':
            self.pos_enc = PosEncoding(args.hidden_dim, args.max_seq_len)
        elif args.pos_enc_type == 'learn':
            # need to modify if you want to use PosLearnedEncoding
            self.pos_enc = PosLearnedEncoding(args.hidden_dim)
        else:
            raise KeyError(f'{args.pos_enc_type} is not supported for positional encoding')

        self.enc_layernorm = nn.LayerNorm(args.hidden_dim)
        if self.args.vis_droput_ratio > 0.0:
            self.dropout_vis = nn.Dropout(args.vis_droput_ratio, inplace=True)
        if args.transinput_drop_ratio > 0.0:
            self.enc_inp_dropout = nn.Dropout(args.transinput_drop_ratio, inplace=True)

        self.generate_attn_mask = getattr(self, f'generate_attention_mask_{self.args.attn_mask_type}')

    def forward(self, emb_frames, emb_actions):
        '''
        pass embedded inputs through embeddings and encode them using a transformer
        '''
        ## create a mask for padded elements
        #length_mask_pad = length_frames_max * (
        #    2 if lengths_actions.max() > 0 else 1)
        #mask_pad = torch.zeros(
        #    (len(emb_frames), length_mask_pad), device=emb_frames.device).bool()
        #for i, (len_f, len_a) in enumerate(zip(lengths_frames, lengths_actions)):
        #    # mask padded frames
        #    mask_pad[i, len_f:length_frames_max] = True
        #    # mask padded actions
        #    mask_pad[i, length_frames_max + len_a:] = True

        frame_seq_len = emb_frames.size(1)
        if self.args.vis_droput_ratio > 0.0:
            emb_frames = self.dropout_vis(emb_frames)
        # encode the inputs
        emb_all = self.encode_inputs(emb_frames, emb_actions)

        # create a mask for attention (prediction at t should not see frames at >= t+1)
        # assert length_frames_max == max(lengths_actions)
        mask_attn = self.generate_attn_mask(self.args.max_seq_len, emb_all.device)

        #print('[debug] mask_attn:\n', mask_attn)
        #print('[debug] mask_attn shape:', mask_attn.shape)

        # encode the inputs
        output = self.enc_transformer(
            # since nn.TransformerEncoderLayer is set as batch_first = False
            # the shape of emb_all has to be modified from (batch, seq, feature)
            # to (seq, batch, feature).
            # At the end of the calculation, the shape is turned back to (batch, seq, feature)
            emb_all.transpose(0, 1), mask_attn).transpose(0, 1)
        # visual embs are only used for prediction
        return output[:, :frame_seq_len, :]

    def test_step(self, emb_frames, emb_actions, test_seq_id):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''
        if self.args.vis_droput_ratio > 0.0:
            emb_frames = self.dropout_vis(emb_frames)
        # encode the inputs
        emb_all = self.encode_inputs(emb_frames, emb_actions)

        #src_key_padding_mask = self.generate_src_key_padding_mask(test_seq_id, emb_all.device)

        #print('[debug] src_key_padding_mask:', src_key_padding_mask)
        #print('[debug] src_key_padding_mask shape:', src_key_padding_mask.shape)

        mask_attn = self.generate_attn_mask(self.args.max_seq_len, emb_all.device)
        # encode the inputs
        output = self.enc_transformer(
            # since nn.TransformerEncoderLayer is set as batch_first = False
            # the shape of emb_all has to be modified from (batch, seq, feature)
            # to (seq, batch, feature).
            # At the end of the calculation, the shape is turned back to (batch, seq, feature)
            #emb_all.transpose(0, 1), mask=mask_attn, src_key_padding_mask=src_key_padding_mask).transpose(0, 1)
            emb_all.transpose(0, 1), mask=mask_attn).transpose(0, 1)
        # last step of visual emb is only passed for test
        if test_seq_id == self.args.max_seq_len: test_seq_id -= 1
        return output[:, test_seq_id, :]

    def encode_inputs(self, emb_frames, emb_actions):
        '''
        add encodings (positional, token and so on)
        '''
        emb_frames, emb_actions = self.pos_enc(emb_frames, emb_actions)
        # concatenate w.r.t seq len
        emb_cat = torch.cat((emb_frames, *emb_actions), dim=1)
        emb_cat = self.enc_layernorm(emb_cat)
        if self.args.transinput_drop_ratio:
            emb_cat = self.enc_inp_dropout(emb_cat)
        return emb_cat

    def generate_attention_mask_8th(self, len_frames, device):
        '''
        generate mask for attention (a timestep at t does not attend to timesteps after t)'''
        # frames should attend to frames with timestep <= t
        frames_to_frames = self.upper_triangular_mask(len_frames, device)
        frames_to_actions1 = frames_to_frames.clone()
        frames_to_actions1 += self.lower_triangular_mask(len_frames, device, self.args.num_action_history)
        # 2 inside of (1, 2) indicates the total number of agents. So need to adjust if you wanna change it.
        frames_to_all_actions = frames_to_actions1.repeat(1, 2)
        frames_to_all = torch.cat([frames_to_frames, frames_to_all_actions], 1)

        all_mask = self.generate_all_mask(len_frames, device)
        all_mask_except_diag = self.generate_all_mask_except_diag(len_frames, device)
        actions1_to_all = torch.cat([all_mask_except_diag, all_mask_except_diag, all_mask], 1)
        actions2_to_all = torch.cat([all_mask_except_diag, all_mask, all_mask_except_diag], 1)
        all_to_all = torch.cat([frames_to_all, actions1_to_all, actions2_to_all], 0)
        return all_to_all

    def generate_attention_mask_7th(self, len_frames, device):
        '''
        generate mask for attention (a timestep at t does not attend to timesteps after t)'''
        # frames should attend to frames with timestep <= t
        frames_to_frames = self.triangular_mask(len_frames, device)
        frames_to_all = frames_to_frames.repeat(1, 3)

        all_mask = self.generate_all_mask(len_frames, device)
        all_mask_except_diag = self.generate_all_mask_except_diag(len_frames, device)
        actions1_to_all = torch.cat([frames_to_frames, all_mask_except_diag, all_mask], 1)
        actions2_to_all = torch.cat([frames_to_frames, all_mask, all_mask_except_diag], 1)
        all_to_all = torch.cat([frames_to_all, actions1_to_all, actions2_to_all], 0)
        return all_to_all

    def generate_attention_mask_6th(self, len_frames, device):
        '''
        generate mask for attention (a timestep at t does not attend to timesteps after t)'''
        # frames should attend to frames with timestep <= t
        frames_to_frames = self.triangular_mask(len_frames, device)
        frames_to_all = frames_to_frames.repeat(1, self.args.num_agents+1)

        all_mask = self.generate_all_mask(len_frames, device)
        actions_to_all = []
        for i in range(self.args.num_agents):
            tmp_list = []
            for j in range(self.args.num_agents+1):
                if j == 0 or i == j-1:
                    tmp_list.append(frames_to_frames)
                else:
                    tmp_list.append(all_mask)
            
            actions_to_all.append(torch.cat(tmp_list, 1))

        all_to_all = torch.cat([frames_to_all, *actions_to_all], 0)
        return all_to_all

    def generate_attention_mask_5th(self, len_frames, device):
        '''
        generate mask for attention (a timestep at t does not attend to timesteps after t)'''
        # frames should attend to frames with timestep <= t
        frames_to_frames = self.triangular_mask(len_frames, device)
        frames_to_all = frames_to_frames.repeat(1, self.args.num_agents+1)

        all_mask = self.generate_all_mask(len_frames, device)
        all_mask_except_diag = self.generate_all_mask_except_diag(len_frames, device)
        actions_to_all = []
        for i in range(self.args.num_agents):
            tmp_list = []
            for j in range(self.args.num_agents+1):
                if j == 0: 
                    tmp_list.append(all_mask_except_diag)
                elif i == j-1:
                    tmp_list.append(all_mask_except_diag)
                else:
                    tmp_list.append(all_mask)
            
            actions_to_all.append(torch.cat(tmp_list, 1))

        all_to_all = torch.cat([frames_to_all, *actions_to_all], 0)
        return all_to_all

    def generate_attention_mask_4th(self, len_frames, device):
        '''
        generate mask for attention (a timestep at t does not attend to timesteps after t)'''
        # frames should attend to frames with timestep <= t
        frames_to_frames = self.upper_triangular_mask(len_frames, device)
        frames_to_actions1 = frames_to_frames.clone()
        frames_to_actions1 += self.lower_triangular_mask(len_frames, device, self.args.num_action_history)
        # 2 inside of (1, 2) indicates the total number of agents. So need to adjust if you wanna change it.
        frames_to_all_actions = frames_to_actions1.repeat(1, 2)
        frames_to_all = torch.cat([frames_to_frames, frames_to_all_actions], 1)

        all_mask = self.generate_all_mask(len_frames, device)
        actions1_to_all = torch.cat([frames_to_frames, frames_to_frames, all_mask], 1)
        actions2_to_all = torch.cat([frames_to_frames, all_mask, frames_to_frames], 1)
        all_to_all = torch.cat([frames_to_all, actions1_to_all, actions2_to_all], 0)
        return all_to_all

    def generate_attention_mask_3rd(self, len_frames, device):
        '''
        generate mask for attention (a timestep at t does not attend to timesteps after t)'''
        # frames should attend to frames with timestep <= t
        frames_to_frames = self.upper_triangular_mask(len_frames, device)
        frames_to_actions1 = frames_to_frames.clone()
        frames_to_actions1 += self.lower_triangular_mask(len_frames, device, self.args.num_action_history)
        # 2 inside of (1, 2) indicates the total number of agents. So need to adjust if you wanna change it.
        frames_to_all_actions = frames_to_actions1.repeat(1, 2)
        frames_to_all = torch.cat([frames_to_frames, frames_to_all_actions], 1)

        all_mask = self.generate_all_mask(len_frames, device)
        all_mask_except_diag = self.generate_all_mask_except_diag(len_frames, device)
        actions1_to_all = torch.cat([frames_to_frames, all_mask_except_diag, all_mask], 1)
        actions2_to_all = torch.cat([frames_to_frames, all_mask, all_mask_except_diag], 1)
        all_to_all = torch.cat([frames_to_all, actions1_to_all, actions2_to_all], 0)
        return all_to_all

    def generate_attention_mask_2nd(self, len_frames, device):
        '''
        generate mask for attention (a timestep at t does not attend to timesteps after t)'''
        # frames should attend to frames with timestep <= t
        frames_to_frames = self.upper_triangular_mask(len_frames, device)
        frames_to_actions1 = frames_to_frames.clone()
        frames_to_actions1 += self.lower_triangular_mask(len_frames, device, self.args.num_action_history)
        # 2 inside of (1, 2) indicates the total number of agents. So need to adjust if you wanna change it.
        frames_to_all_actions = frames_to_actions1.repeat(1, 2)
        frames_to_all = torch.cat([frames_to_frames, frames_to_all_actions], 1)

        all_mask = self.generate_all_mask(len_frames, device)
        all_mask_except_diag = self.generate_all_mask_except_diag(len_frames, device)
        actions1_to_all = torch.cat([all_mask, all_mask_except_diag, all_mask], 1)
        actions2_to_all = torch.cat([all_mask, all_mask, all_mask_except_diag], 1)
        all_to_all = torch.cat([frames_to_all, actions1_to_all, actions2_to_all], 0)
        return all_to_all

    def generate_attention_mask_1st(self, len_frames, device):
        '''
        generate mask for attention (a timestep at t does not attend to timesteps after t)'''
        # frames should attend to frames with timestep <= t
        frames_to_frames = self.triangular_mask(len_frames, device)
        frames_to_all = frames_to_frames.repeat(1, self.args.num_agents+1)
        all_to_all = frames_to_all.repeat(self.args.num_agents+1, 1)
        return all_to_all

    def generate_src_key_padding_mask(self, test_seq_id, device):
        '''
        generate mask for padding (timesteps larger than self.test_seq_id needs to be ignored)'''
        src_key_padding_mask = torch.zeros((1, self.args.max_seq_len), device=device)
        src_key_padding_mask[:, test_seq_id+1:] = 1
        return src_key_padding_mask.repeat(1, 3).bool()

    def triangular_mask(self, size, device, diagonal_shift=1):
        '''
        generate upper triangular matrix filled with ones
        '''
        square = torch.triu(torch.ones(size, size, device=device), diagonal=diagonal_shift)
        square = square.masked_fill(square == 1., float('-inf'))
        return square

    def upper_triangular_mask(self, size, device, diagonal=1):
        '''
        generate upper triangular matrix filled with ones
        '''
        square = torch.triu(torch.ones(size, size, device=device), diagonal=diagonal)
        square = square.masked_fill(square == 1., float('-inf'))
        return square

    def lower_triangular_mask(self, size, device, diagonal):
        '''
        generate lower triangular matrix filled with ones (used for action attention)
        '''
        # -diagonal controls how much attention 
        square = torch.tril(torch.ones(size, size, device=device), diagonal=-diagonal)
        square = square.masked_fill(square == 1., float('-inf'))
        return square

    def generate_all_mask(self, size, device):
        square = torch.ones(size, size, device=device)
        square = square.masked_fill(square == 1., float('-inf'))
        return square

    def generate_all_mask_except_diag(self, size, device):
        square = torch.tril(torch.ones(size, size, device=device), diagonal=-1) + \
                 torch.triu(torch.ones(size, size, device=device), diagonal=1)
        square = square.masked_fill(square == 1., float('-inf'))
        return square


class PosEncoding(nn.Module):
    '''
    Transformer-style positional encoding with wavelets
    '''
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe.shape = (max_len, d_model) -> pe[None].shape = (1, max_len, d_model)
        self.register_buffer('pe', pe[None])

    def forward(self, frames, actions):
        enc = self.pe[:, :frames.shape[1]]
        enc = enc / math.sqrt(self.d_model)
        frames = frames + enc[:, :frames.shape[1]]
        rtn_acts = [actions[i] + enc[:, :frames.shape[1]] for i in range(len(actions))]
        return frames, rtn_acts


class PosLearnedEncoding(nn.Module):
    # TODO: Needs to be modified if you wanna use it
    '''
    Learned additive positional encoding implemented on top of nn.Embedding
    '''
    def __init__(self, d_model, max_pos=1250, init_range=0.1):
        super().__init__()
        self.emb = nn.Embedding(max_pos, d_model)
        self.emb.weight.data.uniform_(-init_range, init_range)

    def forward(self, lang, frames, actions, lens_lang, lens_frames):
        pos_lang = torch.stack([torch.arange(0, lang.shape[1])] * lang.shape[0])
        pos_frames = torch.stack([torch.arange(0, frames.shape[1]) + l for l in lens_lang])
        # use the same position indices for actions as for the frames
        pos_actions = torch.stack([torch.arange(0, actions.shape[1]) + l for l in lens_lang])
        lang += self.emb(pos_lang.to(lang.device))
        frames += self.emb(pos_frames.to(frames.device))
        actions += self.emb(pos_actions.to(actions.device))
        return lang, frames, actions


class TokenTypeEncoding(nn.Module):
    '''
    Learned additive img/action token encoding implemented on top of nn.Embedding
    '''
    def __init__(self, d_model, num_token_type=2, init_range=0.1):
        super().__init__()
        self.emb = nn.Embedding(num_token_type, d_model)
        self.emb.weight.data.uniform_(-init_range, init_range)

    def forward(self, frames_actions, token_type_ids):
        # token_type_ids.shape = (bs, max_seq_len)
        frames_actions += self.emb(token_type_ids)
        return frames_actions


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pong')
    parser.add_argument('--num_agents', type=str, default=None)
    parser.add_argument('--attn_mask_type', type=str, default='5th')
    args = parser.parse_args()
    
    class Opts:
        def __init__(self, 
                     action_space, 
                     num_agents=2, 
                     attn_mask_type='5th', 
                     continuous_action=False):
            self.bs = 2
            self.action_space = action_space
            self.hidden_dim = 512
            self.ff_dim = 512
            self.act_drop_ratio = 0.0
            self.transenc_heads = 8
            self.tranenclayer_drop_ratio = 0.1
            self.num_transenclayer = 2
            self.pos_enc_type = 'original'
            self.max_seq_len = 3
            self.vis_droput_ratio = 0.0
            self.transinput_drop_ratio = 0.0
            self.test = False
            self.num_action_history = 1
            self.attn_mask_type = attn_mask_type
            self.num_agents = num_agents
            self.continuous_action = continuous_action

    print('dataset:', args.dataset)
    continuous_action = False
    if args.dataset == 'pong':
        action_space = 4
        if not args.num_agents: 
            args.num_agents = 2
        else:
            assert args.num_agents == 2 or args.num_agents == 4, 'num_agents for pong \
                                                                has to be either 2 or 4.'
    elif args.dataset == 'gtav':
        action_space = 3
        args.num_agents = 1
    elif args.dataset == 'carla':
        action_space = 2
        args.num_agents = 1
        continuous_action = True
    else:
        assert 0, f'test case {args.dataset} is not implemented yet.'

    opts = Opts(action_space, args.num_agents, args.attn_mask_type, continuous_action)
    print('bs:', opts.bs)
    print('max_seq_len:', opts.max_seq_len)
    print('num_agents:', opts.num_agents)
    print('attn_mask_type:', opts.attn_mask_type)
    emb_frames = torch.randn(opts.bs, opts.max_seq_len, opts.hidden_dim)
    if continuous_action == True:
        actions = [torch.randn((opts.bs, opts.max_seq_len, opts.action_space)) for _ in range(opts.num_agents)]
    else:
        actions = [torch.randint(0, opts.action_space, (opts.bs, opts.max_seq_len))
                                for _ in range(opts.num_agents)]
    print('emb_frams shape:', emb_frames.shape)
    print('action shape:', actions[0].shape)
    model = TransitionLearner(opts)
    print('output shape:', model(emb_frames, actions).shape)