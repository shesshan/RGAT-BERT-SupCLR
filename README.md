# RGAT-BERT-SupCLR
This repo contains the PyTorch implementaion for RGAT-BERT-SupCLR model.

For any questions about the implementation, please email ms.chang@siat.ac.cn.

## Requirements
* Python 3.9.7

* PyTorch 1.11.0

* Transformers https://github.com/huggingface/transformers

* CUDA 11.4

  To install requirements, run `pip install -r requirements.txt`

## Perparation
### For Glove Embedding
First, download and unzip GloVe vectors(`glove.840B.300d.zip`) from https://nlp.stanford.edu/projects/glove/. Then change the value of parameter `--glove_dir` to the directory of the word vector file.

### For BERT Embedding
Download the pytorch version pre-trained `bert-base-uncased` model and vocabulary from the link provided by huggingface. Then change the value of parameter `--bert_model_dir` to the directory of the bert model.

## Data Preprocess
The preprocess codes are in `data_preprocess_xml.py` (for res14, lap14 datasets) and `data_preprocess_raw.py` (for res15, res16, twitter datasets) . However, here already provides the preprocessed datasets with dependency parcing results in `./data/`, so you can skip preprocess.

If you want to preprocess new ABSA datasets, please first download and unzip biaffine parser from https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz. Then change the value of parameter `--model_path` and  `--data_path`  (in  `data_preprocess_xml.py` and `data_preprocess_raw.py`) to the directory of the parser and raw data files.

## Training
Run:
`bash run.sh`
