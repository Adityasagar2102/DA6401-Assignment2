# DA6401 Assignment 2

## Building a Complete Visual Perception Pipeline

**Author:** Aditya Kumar Sagar  
**Roll Number:** CS25M006  
**Course:** DA6401 – Deep Learning  
**Institute:** IIT Madras  

---

## GitHub Repository

* you can find the complete project on: [GitHub](https://github.com/Adityasagar2102/DA6401-Assignment2)

## Weights & Biases Report

* **Public Report Link:** [W&B Report](https://api.wandb.ai/links/cs25m006-iit-madras/3zoe3q3y)

---

## Overview

This project implements a full multi-task visual perception pipeline on the **Oxford-IIIT Pet Dataset**. A single shared VGG11 encoder backbone simultaneously drives three task heads:

- **Classification** — predicts the pet breed (37 classes)
- **Localization** — regresses a bounding box around the pet's head
- **Segmentation** — produces a pixel-wise trimap mask (foreground / background / boundary)

All three tasks share one forward pass through the unified `MultiTaskPerceptionModel`.

---

## Project Structure

```
da6401_assignment_2/
│
├── checkpoints/
│   └── checkpoints.md          # Placeholder; .pth files downloaded via gdown at runtime
│
├── data/
│   └── pets_dataset.py         # OxfordIIITPetDataset — multi-task dataloader
│
├── losses/
│   ├── __init__.py
│   └── iou_loss.py             # Custom IoU loss (nn.Module, no external libraries)
│
├── models/
│   ├── __init__.py
│   ├── layers.py               # CustomDropout (inverted dropout, nn.Module from scratch)
│   ├── vgg11.py                # VGG11Encoder + full VGG11 classifier
│   ├── classification.py       # VGG11Classifier head
│   ├── localization.py         # VGG11Localizer — regression head
│   ├── segmentation.py         # VGG11UNet — U-Net style decoder
│   └── multitask.py            # MultiTaskPerceptionModel — unified pipeline
│
├── wandb/                      # Auto-generated WandB run logs (git-ignored)
├── inference.py                # CLI inference script with visualisation
├── train.py                    # Training entry-point for all three tasks
├── requirements.txt
└── README.md
```

> **Note:** The Oxford-IIIT Pet dataset is **not** included in this repository. The `data/` folder contains only the dataset loader (`pets_dataset.py`). The actual images and annotations must be downloaded and placed inside `data/` as described in the Setup section below.

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/Adityasagar2102/DA6401-Assignment2.git
cd da6401_assignment_2
pip install -r requirements.txt
```

### 2. Download the Oxford-IIIT Pet Dataset

Download and extract the dataset into the `data/` folder so the final layout looks like this:

```
data/
├── pets_dataset.py
├── images/
│   └── *.jpg
└── annotations/
    ├── trimaps/
    │   └── *.png
    └── xmls/
        └── *.xml
```

```bash
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

tar -xvf images.tar.gz      -C data/
tar -xvf annotations.tar.gz -C data/
```

---

## Training

Always train in the order **classify → localize → segment**, because the localizer and segmentation model initialise their encoders from the saved classifier checkpoint.

```bash
# Train all three tasks sequentially (recommended)
python train.py --task all --epochs 50 --batch_size 32 --lr 1e-4

# Or train each task individually
python train.py --task classify  --epochs 50
python train.py --task localize  --epochs 40
python train.py --task segment   --epochs 40
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | `all` | `classify`, `localize`, `segment`, `all`, `dropout_sweep` |
| `--data_dir` | `data` | Root of the pet dataset (contains `images/` and `annotations/`) |
| `--checkpoint_dir` | `checkpoints` | Where to save `.pth` files |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `1e-4` | Learning rate |
| `--weight_decay` | `1e-4` | L2 regularisation |
| `--dropout_p` | `0.5` | Dropout probability for classifier/localizer heads |
| `--wandb_project` | `da6401-assignment2` | WandB project name |

Trained checkpoints are saved to `checkpoints/classifier.pth`, `checkpoints/localizer.pth`, and `checkpoints/unet.pth`.

---

## Inference

```bash
python inference.py \
    --image path/to/your/pet.jpg \
    --save  output_annotated.jpg
```

The script loads all three heads via `MultiTaskPerceptionModel`, runs a single forward pass, and overlays the predicted bounding box and segmentation mask on the original image.

---

## Architecture

### VGG11 Encoder (`models/vgg11.py`)

Implemented from scratch following the original VGG paper (Simonyan & Zisserman, 2014) with two modern additions: **BatchNorm2d** after every convolutional layer and **CustomDropout** in the fully-connected heads.

| Block | Layers | Output channels | Feature map (224×224 input) |
|---|---|---|---|
| Block 1 | Conv → BN → ReLU | 64 | 224 × 224 |
| Block 2 | Conv → BN → ReLU | 128 | 112 × 112 |
| Block 3 | Conv × 2 → BN → ReLU | 256 | 56 × 56 |
| Block 4 | Conv × 2 → BN → ReLU | 512 | 28 × 28 |
| Block 5 | Conv × 2 → BN → ReLU | 512 | 14 × 14 |
| Pool | MaxPool | 512 | 7 × 7 (bottleneck) |

The encoder is used as the shared backbone in the multi-task model. All intermediate feature maps (f1–f5) are exposed via `return_features=True` for U-Net skip connections.

**Design justification:** BatchNorm is placed immediately after each Conv2d and before ReLU to normalise pre-activation distributions, which stabilises training and enables higher learning rates. CustomDropout (p=0.5) is applied only in the fully-connected heads — not in the convolutional backbone — because spatial feature maps already benefit from the implicit regularisation of weight sharing and BatchNorm; applying Dropout to them degrades spatial structure unnecessarily.

### Custom Dropout (`models/layers.py`)

Implements inverted dropout from scratch using `torch.rand` — no use of `nn.Dropout` or `functional.dropout`. During training, each activation is independently zeroed with probability `p` and the remaining activations are scaled by `1/(1-p)` so the expected output magnitude is unchanged. At eval time the layer is a pure identity.

### Custom IoU Loss (`losses/iou_loss.py`)

Implements intersection-over-union loss as `1 - IoU`, with support for `mean`, `sum`, and `none` reduction. Boxes are accepted in `(x_center, y_center, width, height)` pixel-space format and converted to corner coordinates internally. A small `eps` prevents division by zero when a predicted box has zero area. The loss is bounded in `[0, 1]`.

### Object Localization (`models/localization.py`)

The VGG11 encoder feeds into an adaptive average pool (7×7) followed by a three-layer fully-connected regressor. The final layer outputs four raw values which are passed through `sigmoid × 224` to produce `[x_center, y_center, width, height]` in pixel coordinates. Training uses a combination of normalised SmoothL1 loss and the custom IoU loss.

**Encoder strategy:** The encoder is initialised from the trained classifier weights and kept **frozen** during localizer training. This is critical for the multi-task model: `MultiTaskPerceptionModel` uses the classifier encoder for all three heads, so the regressor must learn to predict from that fixed feature distribution. Fine-tuning the localizer encoder would cause feature drift that breaks the shared backbone.

### U-Net Segmentation (`models/segmentation.py`)

A symmetric decoder mirrors the VGG11 encoder using `ConvTranspose2d` for learnable upsampling (bilinear interpolation is not used). At each decoder stage, the upsampled feature map is concatenated with the corresponding encoder skip connection before two rounds of Conv → BN → ReLU. The final `1×1` convolution produces 3-class logits.

**Loss:** Weighted CrossEntropyLoss (pet=2×, background=1×, boundary=5×) combined with a differentiable soft Dice loss. The boundary class receives the highest weight because it is the smallest and most spatially ambiguous class.

**Encoder strategy:** Same as localization — the encoder is frozen so the decoder learns from the same fixed feature space the multi-task model provides.

### Multi-Task Pipeline (`models/multitask.py`)

`MultiTaskPerceptionModel` loads the three saved checkpoints at initialisation (downloading them via `gdown` if needed), extracts the shared encoder from the classifier, then attaches the localizer regressor head and the full U-Net decoder. A single `forward(x)` call runs the encoder once and fans out to all three task heads, returning a dict with keys `classification`, `localization`, and `segmentation`.

---

## Results

| Metric | Value |
|---|---|
| Classification Macro-F1 | `0.9093` |
| Localization Acc @ IoU ≥ 0.5 | `61.2%` |
| Localization Acc @ IoU ≥ 0.75 | `40.0%` |
| Segmentation Macro-Dice | `0.9235` |

---

## WandB Experiments

All training runs are logged to the public WandB project. The report covers:

- **Section 2.1** — Activation distributions of the 3rd and last conv layers, with and without BatchNorm
- **Section 2.2** — Overlaid train/val loss curves for dropout p = 0.0, 0.2, 0.5
- **Section 2.3** — Transfer learning showdown: frozen encoder vs. partial fine-tuning vs. full fine-tuning
- **Section 2.4** — Feature map visualisations from Block 1 and Block 5
- **Section 2.5** — Interactive detection table (10 validation images, GT=green, Pred=red, IoU shown)
- **Section 2.6** — Segmentation samples: original / GT trimap / predicted trimap, with Dice vs pixel accuracy analysis
- **Section 2.7** — Pipeline results on 3 novel internet images
- **Section 2.8** — Meta-analysis: training curves for all tasks, retrospective architectural reflection

> **WandB Report:** `https://api.wandb.ai/links/cs25m006-iit-madras/3zoe3q3y`

---

## Permitted Libraries

As per the assignment specification, only the following libraries are used:

`torch`, `torchvision` (for pretrained weights only), `numpy`, `albumentations`, `opencv-python`, `scikit-learn`, `wandb`, `gdown`

---

## References

- Simonyan, K. & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- Parkhi, O. M. et al. (2012). *Cats and Dogs.* CVPR. [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- Ronneberger, O., Fischer, P. & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
