CUDA_VISIBLE_DEVICES=0 python trans_learner_training_gan.py \
     --log_dir translearner_results/pong/gan/mask_5th/seq_128_layer_24_bs_64_sw \
     --data_dir encoded_dataset/pong/ \
     --num_transenclayer 24 \
     --attn_mask_type 5th \
     --dataset pong \
     --num_agents 2 \
     --max_seq_len 128 \
     --action_space 4 \
     --seq_weights \
     --batch_size 64