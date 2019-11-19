BERT_OWE (KB embedding) lm_finetune 학습 코드

python simple_lm_finetuning.py \
  --train_corpus /hdd/pshy410/AAAI2020XLNet_KB/prepro  \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --output_dir ./finetuned_lm/ \
  --do_train \
  --num_train_epochs 1.0 \
  --on_memory \
  --output_attentions \
  -c /hdd/pshy410/AAAI2020XLNet_KB/dataset/FB15k-237-OWE \
  -d OWE_output \
  --complex /hdd/pshy410/AAAI2020XLNet_KB/dataset/FB15k-237-OWE-closed-world-embeddings/embeddings \
  --load_best


python simple_lm_finetuning.py \
  --train_corpus /hdd/pshy410/AAAI2020XLNet_KB/prepro  \
  --bert_model pretrained_finetuned_lm \
  --do_lower_case \
  --output_dir ./finetuned_lm/ \
  --do_train \
  --num_train_epochs 1.0 \
  --on_memory \
  --output_attentions \
  -c /hdd/pshy410/AAAI2020XLNet_KB/dataset/FB15k-237-OWE \
  -d OWE_output \
  --complex /hdd/pshy410/AAAI2020XLNet_KB/dataset/FB15k-237-OWE-closed-world-embeddings/embeddings \
  --load_best
  
  
=== GLUE 평가
export GLUE_DIR=/hdd/pshy410/AAAI2020XLNet_KB/dataset/GLUE
export TASK_NAME=MNLI

python run_glue.py \
    --model_type bert \
    --model_name_or_path /hdd/pshy410/AAAI2020XLNet_KB/pytorch-transformers_OWE/examples/lm_finetuning/pretrained_finetuned_lm \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir ./tmp/$TASK_NAME/ \
	--eval_all_checkpoints
  
  
학습은 안시키고 평가만 하고싶을때

python run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --output_dir ./tmp/$TASK_NAME/ \
	--eval_all_checkpoints


==================
BERT_OWE (KB embedding) 평가코드

export SWAG_DIR=/hdd/pshy410/swagaf


## --bert_model /hdd/pshy410/AAAI2020XLNet_KB/pytorch-pretrained-BERT_OWE/examples/lm_finetuning/finetuned_lm \
python run_swag.py \
  --bert_model tmp/[0.19]swag_output \
  --do_train \
  --do_lower_case \
  --do_eval \
  --data_dir $SWAG_DIR/data \
  --train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 80 \
  --output_dir ./tmp/swag_output/ \
  --gradient_accumulation_steps 4 \
  -c /hdd/pshy410/AAAI2020XLNet_KB/dataset/FB15k-237-OWE \
  -d lm_finetuning/OWE_output \
  --complex /hdd/pshy410/AAAI2020XLNet_KB/dataset/FB15k-237-OWE-closed-world-embeddings/embeddings \
  --load_best
  

  
python run_swag.py \
  --bert_model /hdd/pshy410/AAAI2020XLNet_KB/pytorch-pretrained-BERT_OWE/examples/tmp/[trained]swag_output \
  --do_lower_case \
  --do_eval \
  --data_dir $SWAG_DIR/data \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 80 \
  --output_dir ./tmp/swag_output/ \
  --gradient_accumulation_steps 4 \
  -c /hdd/pshy410/AAAI2020XLNet_KB/dataset/FB15k-237-OWE \
  -d lm_finetuning/OWE_output \
  --complex /hdd/pshy410/AAAI2020XLNet_KB/dataset/FB15k-237-OWE-closed-world-embeddings/embeddings \
  --load_best
  
  
  
==================================
commonsenseQA 에서 돌릴때

python run_commonsense_qa.py
  --split=$SPLIT \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR