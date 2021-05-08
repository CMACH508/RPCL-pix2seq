# RPCL-pix2seq: Controllable sketch synthesis from a self-organized latent space

Free-hand sketches in the same category can be totally different in *styles*. Here *styles* represents all non-categorical patterns (e.g., a pig sketch with a single head or one with the whole body, a giraffe sketch orienting left or right), which could be caused by drawing manners, conceptual bias, etc. Thus, when controllablly generating a specific sketch, both the categorical and stylistic patterns should be considered. But these *styles* are always **unlabeled**. RPCL-pix2seq aims to unsupervisedly self-organize the sketches into a Gaussian Mixture Model (GMM) structured latent space, where sketches with similar categories and styles are clustered in the same Gaussian component. Moreover, enhanced by Rival Penalized Competitive Learning (RPCL) strategy, RPCL-pix2seq is able to automatically determine an appropriate Gaussian number in the GMM structure, making the controllable synthesis robust. 

This repo contains the TensorFlow code for `RPCL-pix2seq`, and more information can be found in [Controllable stroke-based sketch synthesis from a self-organized latent space](https://www.sciencedirect.com/science/article/abs/pii/S0893608021000149).

# Overview

RPCL-pix2seq is a generative model for stroke-based free-hand sketch in a hierarchical structure under the Variational Auto-Encoder (VAE) framework. The bottom layer contains a CNN encoder for feature extraction, an RNN decoder (directly adopted from [sketch-rnn](https://github.com/magenta/magenta/tree/master/magenta/models/sketch_rnn)) for sketch synthesis and an CNN decoder as a regularization. And the top layer is a rival penalized EM-like algorithm for learning a GMM-structured latent space. 

<img src="https://github.com/CMACH508/RPCL-pix2seq/blob/main/assets/RPCL-pix2seq.png" width="650" alt="overview"/>

During training, RPCL-pix2seq firstly extracts a latent code `batch_z` (with a dimension of `z_size`) for the fed sketch image. Based on the latent code, the top layer estimates and then updates the GMM parameters (`de_alpha`, `de_mu`, `de_sigma2` in `model.py`) of the latent space. Then the code is respectively sent into the two-branches decoders, generating a series of pen strokes as the sequence-formed sketch and a pixel-formed image reconstruction. The pixel-formed image reconstruction from the CNN decoder works as a regularization, encouraging the encoder to preserve more features from the fed sketch image to the latent code.

<img src="https://github.com/CMACH508/RPCL-pix2seq/blob/main/assets/latent_space.jpg" width="650" alt="latent_space"/>

Sketches with different categorical and stylistic patterns are automatically partitioned into clusters in the latent space. When the GMM latent space is initialized with 10 Gaussians, RPCL-pix2seq is able to unsupervisedly self-organize a GMM space with 7 Gaussian components, according to the training dataset. The redundant 3 Gaussians (Gaussian #6, #8, #9 in the figure above) are automatically kicked out by RPCL-pixseq itself during training. With an appropriate number of Guassian components left, the latent space is smooth enough to do synthesis reasoning (such as interpolation in below) generating novel but reasonable sketches which neither appear in the training dataset nor exist in real life.

<img src="https://github.com/CMACH508/RPCL-pix2seq/blob/main/assets/interpolation_2d.png" width="500" alt="interpolation"/>

# Training an RPCL-pix2seq

## Dataset

Before training an RPCL-pix2seq, you first need a pixel-formed sketch dataset translated from [QuickDraw dataset](https://quickdraw.withgoogle.com/data). Each sketch image is in **48x48x1**. The provided `seq2png.py` is used to create the required dataset. You are able to build your own pixel-formed dataset based on QuickDraw dataset with
``python seq2png.py``, and it follows an example usage.

```
python seq2png.py --input_dir=quickdraw_path --output_dir=png_path --png_width=48 --categories={'bee','bus'}
```
Each category of sketch images will be packaged in a single `.npz` file, and it will take about 30 to 60 minutes for each file translation. You might need the `svgwrite` python module, which can be installed as

```
conda install -c omnia svgwrite=1.1.6
```

## Required environments

1. Python 3.6
2. Tensorflow 1.12

## Training
```
python train.py --log_root=checkpoint_path --data_dir=dataset_path --resume_training=False --hparams="categories=[bee,bus], dec_model=hyper, batch_size=128"
```

`checkpoint_path` and `dataset_path` denote the model saving dir and the dataset dir, respectively. For the `hparams`, we provide a list of full options for training RPCL-pix2seq, along with the default settings:
```
categories=['bee', 'bus'],         # Sketch categories for training
num_steps=1000001,                 # Number of total steps (the process will stop automatically if the loss is not improved)
save_every=1,                      # Number of epochs per checkpoint creation
dec_rnn_size=2048,                 # Size of decoder
dec_model='hyper',                 # Decoder: lstm, layer_norm or hyper
max_seq_len=-1,                    # Max sequence length. Computed by DataLoader
z_size=128,                        # Dimension of latent code
batch_size=128,                    # Minibatch size
num_mixture=5,                     # Recommend to set to the number of categories
learning_rate=0.001,               # Learning rate
decay_rate=0.9999,                 # Learning rate decay per minibatch.
min_learning_rate=0.00001,         # Minimum learning rate
grad_clip=1.,                      # Gradient clipping
de_weight=0.5,                     # Weight for deconv loss
use_recurrent_dropout=True,        # Dropout with memory loss
recurrent_dropout_prob=0.90,       # Probability of recurrent dropout keep
use_input_dropout=False,           # Input dropout
input_dropout_prob=0.90,           # Probability of input dropout keep
use_output_dropout=False,          # Output droput
output_dropout_prob=0.9,           # Probability of output dropout keep
random_scale_factor=0.10,          # Random scaling data augmention proportion
augment_stroke_prob=0.10,          # Point dropping augmentation proportion
png_scale_ratio=0.98,              # Min scaling ratio
png_rotate_angle=0,                # Max rotating angle (abs value)
png_translate_dist=0,              # Max translating distance (abs value)
is_training=True,                  # Training mode or not
png_width=48,                      # Width of input sketch images
num_sub=2,                         # Init number of components for each category
num_per_category=70000             # Training samples from each category
```

We also provide a pretrained RPCL-pix2seq model and you can get it from [linkA](https://jbox.sjtu.edu.cn/l/SFjd6L) or [linkB](https://1drv.ms/u/s!ArRhVHnSla1gs1b3HFGVpNf7vKOr?e=BUykFQ).

*Tips: When dealing with a multi-categorized dataset, enlarging the learning rate \eta for latent GMM learning helps accelerate the model selection mechanism.*

## Generating
```
python sample.py --model_dir=checkpoint_path --output_dir=output_path --num_per_category=300 --conditional=True
```

With a trained model, you can generate sketches with sketch images input as conditions or not. For the `conditional` mode, the category and style of the generated sketches each are conditional on a sketch reference, which is randomly selected from the test set. And for the `unconditional` mode, the generated sketch corresponds to the latent code you set in `sample.py` before starting.

`num_per_category` denotes the number of generated sketches per category, and the generated sketches with their corresponding latent codes and Gaussian component indexes are stored in `output_path`. 

## Evaluating
```
python retrieval.py --model_dir=checkpoint_path --sample_dir=output_path
```

The metrics **Rec** and **Ret** can evaluate whether the generated sketches are categorically and stylistically controllable. You need to train a [Sketch_a_net](https://arxiv.org/pdf/1501.07873.pdf) with the same training set as for RPCL-pix2seq, to calculate **Rec**. And you can use `retrieval.py` to get **Ret**. 

<img src="https://github.com/CMACH508/RPCL-pix2seq/blob/main/assets/criteria.png" width="650" alt="criteria"/>

* Please make sure the generated sketches for evaluation are *black-and-white*, and both metrics are calculated with the entire test set (i.e., --num_per_category=2500).


# Citation
If you find this project useful for academic purposes, please cite it as:
```
@Article{RPCL-pix2seq,
  Title                    = {Controllable stroke-based sketch synthesis from a self-organized latent space},
  Author                   = {Sicong Zang and Shikui Tu and Lei Xu},
  Journal                  = {Neural Networks},
  Year                     = {2021},
  Pages                    = {138-150},
  Volume                   = {137},
  Doi                      = {https://doi.org/10.1016/j.neunet.2021.01.006},
  ISSN                     = {0893-6080}
}
```
