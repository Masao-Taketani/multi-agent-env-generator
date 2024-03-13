CUDA_VISIBLE_DEVICES=0 python trans_learner_training_gan.py \
    --log_dir translearner_results/pong_4agents/gan/mask_5th/seq_128_layer_24_bs_31_sw \
    --data_dir encoded_dataset/pong_4agents/ \
    --num_transenclayer 24 \
    --attn_mask_type 5th \
    --dataset pong \
    --num_agents 4 \
    --max_seq_len 128 \
    --batch_size 31 \
    --seq_weights