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
We already preprocess the datasets for you. To be able to compare with the SOTA models,
we adapt the preprocess pipeline right from this previous [repo](https://github.com/HSLCY/ABSA-BERT-pair) where SOTA models are trained. To regenerate the dataset, please refer
to their paper and generate. Please also consider to cite their paper for this process.
|            |                                                                                                                                  PerSent V1                                                                                                                                  |                                                                                                                                                             PerSent V2                                                                                                                                                             |
|:----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  QACG-BERT |                train_para_4aspects_128tokens.csv<br />train_para_7aspects_128tokens.csv<br /> dev_para_4aspects_128tokens.csv<br />dev_para_7aspects_128tokens.csv<br /> random_test_para_4aspects_128tokens.csv<br />random_test_para_7aspects_128tokens.csv                |                                                 train_para_4topics_noNeuMix.csv<br />train_para_7topics_noNeuMix.csv<br /> dev_para_4topics_noNeuMix.csv<br />dev_para_7topics_noNeuMix.csv<br /> random_test_para_4topics_noNeuMix.csv<br />random_test_para_7topics_noNeuMix.csv                                                 |
| Longformer | train_longformer_4topics_auxiliary.csv<br />train_longformer_7topics_auxiliary.csv<br /> dev_longformer_4topics_auxiliary.csv<br />dev_longformer_7topics_auxiliary.csv<br /> random_test_longformer_4topics_auxiliary.csv<br />random_test_longformer_7topics_auxiliary.csv | train_longformer_4topics_auxiliary_noNeuMix.csv<br />train_longformer_7topics_auxiliary_noNeuMix.csv<br /> dev_longformer_4topics_auxiliary_noNeuMix.csv<br />dev_longformer_7topics_auxiliary_noNeuMix.csv<br /> random_test_longformer_4topics_auxiliary_noNeuMix.csv<br />random_test_longformer_7topics_auxiliary_noNeuMix.csv |
|  QACG-LONG |                               train_longformer_4topics.csv<br />train_longformer_7topics.csv<br /> dev_longformer_4topics.csv<br />dev_longformer_7topics.csv<br /> random_test_longformer_4topics.csv<br />random_test_longformer_7topics.csv                               |                                                                train_4topics_noNeuMix.csv<br />train_7topics_noNeuMix.csv<br /> dev_4topics_noNeuMix.csv<br />dev_7topics_noNeuMix.csv<br /> random_test_4topics_noNeuMix.csv<br />random_test_7topics_noNeuMix.csv                                                                |

### Train CG-BERT Model and QACG-BERT Models
Our (T)ABSA BERT models are adapted from [huggingface](https://github.com/huggingface/transformers) BERT model for text classification. If you want to take a look at the original model please search for [BertForSequenceClassification](https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py). To train QACG-BERT model with semeval2014 dataset on GPU 0 and 1, you can do something like this,
```bash
cd code/
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier.py \
--task_name sentihood_NLI_M \
--data_dir ../datasets/sentihood/ \
--output_dir ../results/sentihood/QACGBERT-reproduce/ \
--model_type QACGBERT \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 64 \
--eval_batch_size 256 \
--learning_rate 2e-5 \
--num_train_epochs 30 \
--vocab_file ../models/BERT-Google/vocab.txt \
--bert_config_file ../models/BERT-Google/bert_config.json \
--init_checkpoint ../models/BERT-Google/pytorch_model.bin \
--seed 123 \
--evaluate_interval 25
```
Please take a look at ``code/util/args_parser.py`` to find our different arguments you can pass with. And you can alsp take a look at ``code/util/processor.py`` to see how we process different datasets. We currently supports almost 10 different dataset loadings. You can create your own within 1 minute for loading data. You can specify your directories info above in the command.

### Analyze Attention Weights, Relevance and More
Once you have your model ready, save it to a location that you know (e.g., ``../results/semeval2014/QACGBERT/checkpoint.bin``). Our example code how to get relevance scores is in a jupyter notebook format, which is much easier to read. This is how you will open it,
```bash
cd code/notebook/
jupyter notebook
```
Inside ``visualization``, we provide an example on how to extract attention scores, gradient sensitivity scores!
