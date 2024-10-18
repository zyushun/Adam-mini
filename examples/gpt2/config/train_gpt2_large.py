
# these make the total batch size be ~0.5M
# 6 batch size * 1024 block size * 10 gradaccum * 8 GPUs = 491,520
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 120

n_layer = 36
n_head = 20
n_embd = 1280
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False

max_iters = 30800 # by Chinchilla's law: #tokens = 20 * # model parameters
lr_decay_iters = 30800

# eval stuff
eval_interval = 20
eval_iters = 200 # how many samples used for calulating validation loss
log_interval = 10
ckpt_interval = 1000

# optimizer
algorithm = 'adam_mini'
learning_rate = 2e-4 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 308 # 1% total steps
min_lr = 1e-5 


comment = 'gpt2_large'
save_dir = 'log_gpt2/'+comment
out_dir = 'out-gpt2/' +comment # save ckpt
