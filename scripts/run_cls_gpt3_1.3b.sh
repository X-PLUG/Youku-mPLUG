exp_name='cls/cls_gpt3_1.3B_youku_v0_sharp_2'
PYTHONPATH=$PYTHONPATH:./ \
python -m torch.distributed.launch --nproc_per_node=8 --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --nnodes=$WORLD_SIZE \
  --node_rank=$RANK \
  --use_env downstream/run_cls_distributed_gpt3.py \
  --config ./configs/${exp_name}.yaml \
  --output_dir ./output/${exp_name} \
  --enable_deepspeed \
  --resume path/to/1_3B_mp_rank_00_model_states.pt \
  --bf16
  2>&1 | tee ./output/${exp_name}/train.log
# For testing, set nproc_per_node=1, add "--evaluate_only", and change the 'resume' option to the corresponding checkpoint.
# Also, set model_parallel_size=1 and tensor_model_parallel_size=1 in the corresponding config file.