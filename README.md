# QACG-LONG

```bash
cd code/
CUDA_VISIBLE_DEVICES=0 python3 run_classifier.py \
--task_name persent \
--data_dir ../datasets/persent/ \
--output_dir ../results/persent/QACGLONG-reproduce/ \
--model_type QACGLONG \
--do_lower_case \
--max_seq_length 2048 \
--train_batch_size 8 \
--eval_batch_size 10 \
--learning_rate 1e-4 \
--num_train_epochs 30 \
--vocab_file Longformer/vocab.json \
--bert_config_file Longformer/config.json \
--init_checkpoint Longformer/pytorch_model.bin \
--seed 123 \
--evaluate_interval 125 \
--gradient_accumulation_steps 8
```
