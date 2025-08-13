NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --main_process_port 26902 --num_processes=1 ./train_dc.py --config-name train
