CUDA_VISIBLE_DEVICES=0 python trans_learner_training_gan.py \
    --log_dir translearner_results/boxing/gan/mask_5th/seq_64_layer_24_sw \
    --data_dir encoded_dataset/boxing/ \
    --num_transenclayer 24 \
    --attn_mask_type 5th \
    --dataset boxing \
    --num_agents 2 \
    --max_seq_len 64 \
    --action_space 6 \
    --seq_weights