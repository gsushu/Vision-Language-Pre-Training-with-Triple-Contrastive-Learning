# Vision-Language Pre-Training with Triple Contrastive Learning

This repository contains the implementation and setup instructions for **Vision-Language Pre-Training using Triple Contrastive Learning**, designed to enhance cross-modal representation learning using contrastive objectives between image, text, and joint embeddings.

---

## 🔧 Requirements

Ensure the following are installed in your Python 3 environment:

- Python 3.x
- PyTorch, Torchvision, Torchaudio, CUDAToolkit
- Additional libraries: 
  - `transformers`
  - `timm`
  - `ruamel.yaml`
  - `einops`
- (Optional but recommended) NVIDIA Apex for mixed precision training

Install dependencies via:
pip install -r requirements.txt

Pre-trained Models
Download pre-trained models for initializing encoders:

Image Encoder:

clip-vit-base

swin-transformer-base

Text Encoder:

bert-base

These can be downloaded from Hugging Face or loaded automatically if using transformers.

Pre-training Datasets

| Dataset                  | Download Link                                                                  |
| ------------------------ | ------------------------------------------------------------------------------ |
| COCO                     | [cocodataset.org](https://cocodataset.org/#download)                           |
| Visual Genome (VG)       | [visualgenome.org](https://visualgenome.org/api/v0/api_home.html)              |
| Conceptual Captions (CC) | [google research](https://ai.google.com/research/ConceptualCaptions/download)  |
| SBU Captions             | [Use `img2dataset`](https://github.com/rom1504/img2dataset) with provided URLs |
| CC12M                    | Use `img2dataset` on `cc12m.tsv`                                               |

Downstream Task Datasets
| Task            | Dataset                               | Download Link                                                             |
| --------------- | ------------------------------------- | ------------------------------------------------------------------------- |
| VQA             | Visual Question Answering             | [visualqa.org](https://visualqa.org/download.html)                        |
| NLVR2           | Natural Language for Visual Reasoning | [lil.nlp.cornell.edu/nlvr](https://lil.nlp.cornell.edu/nlvr/)             |
| Image Retrieval | Flickr Image Dataset                  | [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) |

Dataset Setup
Download raw images from the dataset links above.

Retrieve and use the provided .json files containing:

Image paths

Captions or textual descriptions

Ensure correct dataset paths are specified inside the pretrain.py script.

Running the Code
Pre-training
python pretrain.py --config configs/pretrain.yaml
Ensure all datasets and pre-trained models are in place.

Load dataset JSON files into the script.

Training checkpoints will be automatically saved and can be used for downstream tasks.

Downstream Task Evaluation
VQA: python VQA.py

NLVR2: python NLVR2.py

These scripts use the pre-trained model checkpoints to assess task-specific performance.

Notes
Verify all paths and configuration in YAML or JSON files before execution.

For optimal performance, use a GPU-enabled machine with CUDA.

Fine-tuning scripts are included to further improve accuracy and cross-modal alignment.

License
This project is released under the MIT License. See the LICENSE file for details.


