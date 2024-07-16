conda activate rlhf

export HF_HOME="~/.cache/huggingface"

# DeepSpeed Team
BASE_PATH="./"
DATA_PATH="./data"

ACTOR_MODEL_PATH="./sft-checkpoint/"
REWARD_MODEL_PATH="./reward-checkpoint/"
ACTOR_ZERO_STAGE=2
REFERENCE_ZERO_STAGE=3
REWARD_ZERO_STAGE=3

LOG_PATH="${BASE_PATH}/log"
SEED=1234
LR=2e-7
ALGORITHM="adam_mini"

OUTPUT="$LOG_PATH/step3_remax-llama2_7b"

mkdir -p $OUTPUT

ACTOR_LR=$LR

deepspeed --master_port 12341  ../../../step3_remax_finetuning/main.py \
   --algo "remax" \
   --data_path $DATA_PATH  \
   --data_output_path "${BASE_PATH}/tmp" \
   --data_split 4,6,0 \
   --prompt_data_source "reward" \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --reward_model_name_or_path $REWARD_MODEL_PATH \
   --actor_tokenizer_path "meta-llama/Llama-2-7b-hf" \
   --reward_tokenizer_path "meta-llama/Llama-2-7b-hf" \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 12 \
   --per_device_training_batch_size 12 \
   --per_device_eval_batch_size 12 \
   --generation_batches 1 \
   --max_answer_seq_len 512 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate ${ACTOR_LR} \
   --actor_weight_decay 0.0 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --actor_dropout 0.0 \
   --reward_dropout 0.0 \
   --optimizer $ALGORITHM \
   --offload_reference_model \
   --offload_reward_model \
   --num_warmup_steps 10 \
   --penalty "kl_onestep" \
   --kl_ctl 0.1 \
   --deepspeed \
   --dtype bf16 \
   --seed $SEED \
   --data_seed 1234 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --reference_zero_stage $REFERENCE_ZERO_STAGE \
   --reward_zero_stage $REWARD_ZERO_STAGE \
   --enable_hybrid_engine \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_answers \
   --save_answers \
   --eval_interval 20 \
   --eval_samples 500 \
   --flash_attn \
   --save_model \
   &> $OUTPUT/training.log  \
