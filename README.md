Vision-Language Pre-Training with Triple Contrastive Learning
This repository contains the code and instructions for the Vision-Language Pre-Training project using Triple Contrastive Learning. Below are the requirements, dataset details, and steps to set up and run the project.
Requirements
To execute the code, ensure the following are installed:

Python 3 environment
Frameworks: pytorch, torchvision, torchaudio, cudatoolkit
Additional libraries: transformers, timm, ruamel.yaml, einops
For pre-training scripts, install Apex for mixed precision training

All required frameworks are listed in the requirements.txt file. Install them using:
pip install -r requirements.txt

Pre-trained Models
Download the following pre-trained models for parameter initialization:

Image Encoder: clip-vit-base or swin-transformer-base
Text Encoder: bert-base

Datasets
The project uses the following datasets for pre-training and downstream tasks:
Pre-training Datasets

COCO: Download from https://cocodataset.org/#download
Visual Genome (VG): Download from https://visualgenome.org/api/v0/api_home.html
Conceptual Captions (CC): Download from https://ai.google.com/research/ConceptualCaptions/download
SBU Captions:
Obtain URLs from the SBU Captions dataset
Use img2dataset to download images from the URLs


CC12M:
Download the dataset from cc12m.tsv
Use img2dataset to download images from the TSV file



Downstream Task Datasets

VQA: Download from https://visualqa.org/download.html
NLVR2: Download from https://lil.nlp.cornell.edu/nlvr/
Flickr Image Dataset: Download from https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

Dataset Setup

Download raw images from the respective websites.
Download the provided JSON files containing image paths and captions.
Load the JSON files into the pretrain.py script for pre-training.

Running the Code
Pre-training

Setup: Ensure all datasets and pre-trained models are downloaded.
Load Datasets: Input the JSON files with image paths and captions into the pretrain.py script.
Run Pre-training: Execute the pretrain.py script to train the model.
Checkpoints: Evaluate model performance using zero-shot and fine-tuned results on COCO and Flickr30 datasets.

Downstream Tasks

Evaluation Scripts: Use VQA.py and NLVR2.py to evaluate performance on downstream tasks.
Datasets: Run evaluations using COCO and Flickr30 datasets.
Performance Improvement: Enhance accuracy and cross-modal alignment by running the pretrain.py script with the provided fine-tuning code.

Notes

Ensure all dependencies are installed and compatible with your Python environment.
Verify dataset paths in the JSON files before running the scripts.
For optimal performance, use a GPU with CUDA support for cudatoolkit.
