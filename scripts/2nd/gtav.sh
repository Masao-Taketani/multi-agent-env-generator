CUDA_VISIBLE_DEVICES=0 python trans_learner_training_gan.py \
    --log_dir translearner_results/gtav/gan/mask_5th/seq_128_layer_24_sw \
    --data_dir encoded_dataset/gtav/ \
    --num_transenclayer 24 \
    --attn_mask_type 5th \
    --dataset gtav \
    --num_agents 1 \
    --max_seq_len 128 \
    --action_space 3 \
    --seq_weights