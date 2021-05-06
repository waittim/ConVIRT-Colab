# Colab Implement: ConVIRT model
### Contrastive VIsual Representation Learning from Text

The repo is a Colab implementation of the architecture descibed in the ConVIRT paper: [*Contrastive Learning of Medical Visual Representations from Paired Images and Text*](https://arxiv.org/abs/2010.00747). The authors of paper are Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D. Manning, Curtis P. Langlotz.

Deep neural networks learn from a large amount of data to obtain the correct parameters to perform a speciﬁc task. However, in practice, we often encounter a problem: **insuﬃcient amount of labeled data**. However, if your data contains pairs of images and text, you can solve the problem with Contrastive Learning. 

Based on this repository, we can implement various paired-image-text Contrastive Learning tasks on [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb), which enable you to train effective pre-training models for transfer learning with insufficient data volume. With this pre-trained model, you can train with less labeled data to get a good performing model.



## Usage

### 1. Data Preparation

Before starting training, we need to download the training data and make them can be read in pairs. 

There are two example of data preparation:
- Local based: [**data_prepare_MIMIC.ipynb**](https://github.com/waittim/ConVIRT-Colab/blob/master/data_prepare_MIMIC.ipynb)
- Colab based: [**data_prepare_openi.ipynb**](https://github.com/waittim/ConVIRT-Colab/blob/master/data_prepare_openi.ipynb)

After preparation, there should be a CSV file which contains image path and text file path for each paired-image-text. (Or we can save the text content in the CSV file directly.)

### 2. Define Configuration

In **config.yaml**, we need to define the training hyperperemeter, the data path, and the base models. Here is an example:

```
batch_size: 32
epochs: 1000
eval_every_n_epochs: 5
fine_tune_from: Jan16_02-27-36_edu-GPU-Linux
log_every_n_steps: 2
learning_rate: 1e-4
weight_decay: 1e-6
fp16_precision: True
truncation: True

model:
  out_dim: 512
  res_base_model: "resnet50"
  bert_base_model: 'emilyalsentzer/Bio_ClinicalBERT'
  freeze_layers: [0,1,2,3,4,5]
  do_lower_case: False
  
dataset:
  s: 1
  input_shape: (224,224,3)
  num_workers: 4
  valid_size: 0.1
  csv_file: 'path/for/CSV_containing_MIMIC-CXR_paths_for_images_and_text.csv'
  text_from_files: True # If 'True' the text input will be read from .txt files, if 'False' it will be loaded direct from the CSV File 
  img_root_dir: '/your/root/images/directory'
  text_root_dir: '/your/root/text/directory' # The root directory for the text files if "text_from_files" is True
  img_path_col: 0 # index for the image path column in the CSV dataframe.
  text_col: 1 # index for the text column in the CSV dataframe. If text_from_files is 'True' it should contain the relative path for the files from the 'text_root_dir', if text_from_files is 'False' this column should contain the respective input text in its own cells.

loss:
  temperature: 0.1
  use_cosine_similarity: True
  alpha_weight: 0.75
```

The models used (res_base_model, bert_base_model) refers to the models provided by [transformers](https://huggingface.co/transformers/).

### 3. Training

For training in the Colab, please open [**Setup.ipynb**](https://github.com/waittim/ConVIRT-Colab/blob/master/Setup.ipynb), then follow the introduction inside.

After run the code `python run.py` in the notebook, you can open another notebook [**tensorboard.ipynb**](https://github.com/waittim/ConVIRT-Colab/blob/master/tensorboard.ipynb) to monitor the training process.

### 4. After Training

At the end of training, the final model and the corresponding config.yaml will be saved to `./runs/`. Please use this model for transfer learning.


## Others

Note: This repository was forked and modified from https://github.com/sthalles/SimCLR.

References: 
- Yuhao Zhang et al. Contrastive Learning of Medical Visual Representations from Paired Images and Text. https://arxiv.org/pdf/2010.00747.pdf
- Ting Chen et al. A Simple Framework for Contrastive Learning of Visual Representations. https://arxiv.org/abs/2002.05709
- https://github.com/sthalles/SimCLR
- https://github.com/google-research/simclr
- https://github.com/google-research/bert
- https://github.com/hanxiao/bert-as-service#q-what-are-the-available-pooling-strategies
