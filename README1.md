# QACG-LONG
Codebase for integrated model QACG-LONG that is capable to perform targeted aspect-based sentiment analysis at document level. The code is imported from QACG-BERT's open-source code and HuggingFace library's open-source code for Longformer model.

## Quick start
### Download Pretrained BERT Model
You will have to download pretrained BERT model in order to execute the fine-tune pipeline. We recommand to use models provided by the official release on BERT from [BERT-Base (Google's pre-trained models)](https://github.com/google-research/bert). Note that their model is in tensorflow format. To convert tensorflow model to pytorch model, you can use the helper script to do that. For example,
```bash
cd code/
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path uncased_L-12_H-768_A-12/pytorch_model.bin
```

### Datasets
The processed dataset from two versions of PerSent dataset that could be applied to each model directly are listed in the following table. The process of generating each dataset could be referred to ``code/PerSent V1 generation.ipynb`` and ``code/PerSent V2 generation.ipynb``.
|            |                                                                                                                                  PerSent V1                                                                                                                                  |                                                                                                                                                             PerSent V2                                                                                                                                                             |
|:----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  QACG-BERT |                train_para_4aspects_128tokens.csv<br />train_para_7aspects_128tokens.csv<br /> dev_para_4aspects_128tokens.csv<br />dev_para_7aspects_128tokens.csv<br /> random_test_para_4aspects_128tokens.csv<br />random_test_para_7aspects_128tokens.csv                |                                                 train_para_4topics_noNeuMix.csv<br />train_para_7topics_noNeuMix.csv<br /> dev_para_4topics_noNeuMix.csv<br />dev_para_7topics_noNeuMix.csv<br /> random_test_para_4topics_noNeuMix.csv<br />random_test_para_7topics_noNeuMix.csv                                                 |
| Longformer | train_longformer_4topics_auxiliary.csv<br />train_longformer_7topics_auxiliary.csv<br /> dev_longformer_4topics_auxiliary.csv<br />dev_longformer_7topics_auxiliary.csv<br /> random_test_longformer_4topics_auxiliary.csv<br />random_test_longformer_7topics_auxiliary.csv | train_longformer_4topics_auxiliary_noNeuMix.csv<br />train_longformer_7topics_auxiliary_noNeuMix.csv<br /> dev_longformer_4topics_auxiliary_noNeuMix.csv<br />dev_longformer_7topics_auxiliary_noNeuMix.csv<br /> random_test_longformer_4topics_auxiliary_noNeuMix.csv<br />random_test_longformer_7topics_auxiliary_noNeuMix.csv |
|  QACG-LONG |                               train_longformer_4topics.csv<br />train_longformer_7topics.csv<br /> dev_longformer_4topics.csv<br />dev_longformer_7topics.csv<br /> random_test_longformer_4topics.csv<br />random_test_longformer_7topics.csv                               |                                                                train_4topics_noNeuMix.csv<br />train_7topics_noNeuMix.csv<br /> dev_4topics_noNeuMix.csv<br />dev_7topics_noNeuMix.csv<br /> random_test_4topics_noNeuMix.csv<br />random_test_7topics_noNeuMix.csv                                                                |

### Train QACG-LONG
To train QACG-LONG model, you could utilize the following code snippet
```bash
cd code/
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier.py \
--task_name persentv1_para \
--data_dir ../datasets/persent/ \
--output_dir ../results/persent_annotated/QACGLONG-reproduce1/ \
--model_type QACGBERT \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 64 \
--eval_batch_size 256 \
--learning_rate 1e-4 \
--num_train_epochs 30 \
--vocab_file BERT-Google/vocab.txt \
--bert_config_file BERT-Google/config.json \
--init_checkpoint BERT-Google/pytorch_model.bin \
--seed 123 \
--evaluate_interval 25
```
Please take a look at ``code/util/args_parser.py`` to find our different arguments you can pass with. And you can alsp take a look at ``code/util/processor.py`` to see how we process different datasets, remember to change using dataset accordingly for the used processor.
For different argement settings of different cases for the three models, you could refer to the ``code/Trial Run.ipynb``.

### Analyze Attention Weights, Relevance and More
Once you have your model ready, save it to a location that you know (e.g., ``../results/persent_annotated/QACGLONG-reproduce1/``). And the ``code/Evaluation.ipynb`` could be used to analyze the results by using certain models.
