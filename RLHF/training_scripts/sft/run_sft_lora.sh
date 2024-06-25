conda activate rlhf

export HF_HOME="~/.cache/huggingface"

# DeepSpeed Team
BASE_PATH="./"
DATA_PATH="./data"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
ZERO_STAGE=2

LOG_PATH="${BASE_PATH}/log"
SEED=1234
LR=2e-6
ALGORITHM="adamw"

OUTPUT="${LOG_PATH}/step1_sft-llama2_7b-LORA"

mkdir -p $OUTPUT

deepspeed  --master_port 26700 ../../step1_supervised_finetuning/main.py \
   --data_path $DATA_PATH \
   --data_output_path "${BASE_PATH}/tmp" \
   --data_split 4,6,0 \
   --model_name_or_path $MODEL_NAME \
   --tokenizer_path "meta-llama/Llama-2-7b-hf" \
   --per_device_train_batch_size 20 \
   --per_device_eval_batch_size 20 \
   --max_seq_len 1024 \
   --learning_rate $LR \
   --lora_learning_rate $LR \
   --only_optimize_lora \
   --weight_decay 0 \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --lora_dim 128 \
   --num_warmup_steps 10 \
   --seed $SEED \
   --lora_module_name model.layers.\
   --zero_stage $ZERO_STAGE \
   --optimizer $ALGORITHM \
   --deepspeed \
   --dtype bf16 \
   --gradient_checkpointing \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_loss \
   --flash_attn \
   --save_model \
   --eval_interval 20 \
   &> $OUTPUT/training.log
