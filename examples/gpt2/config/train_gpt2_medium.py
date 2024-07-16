# these make the total batch size be ~0.5M
# 6 batch size * 1024 block size * 10 gradaccum * 8 GPUs = 491,520
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 30

n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False

max_iters = 100000
lr_decay_iters = 100000

# eval stuff
eval_interval = 100
eval_iters = 200 # how many samples used for calulating validation loss
log_interval = 10
ckpt_interval = 1000

# optimizer
algorithm = 'adam_mini'
learning_rate = 3e-4 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
epsilon = 1e-8
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
min_lr = 6e-5 


comment = 'gpt2_medium' 
save_dir = 'log_gpt2/'+comment
out_dir = 'out-gpt2/' +comment # save ckpt
