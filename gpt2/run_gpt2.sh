
conda activate gpt2

torchrun --standalone --nproc_per_node=4 --master_port 12345  train_gpt2.py config/train_gpt2_small.py