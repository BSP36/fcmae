# FCMAE-torch

A minimal and flexible PyTorch implementation of the Fully Convolutional Masked AutoEncoder (FCMAE), designed to work **without** MinkowskiEngine and supporting non-square images.

## Features

- 🚀 Pure PyTorch implementation—no MinkowskiEngine required
- 🖼️ Handles non-square images out of the box
- 🛠️ Simple, readable, and easy to extend

## Getting Started

### Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Or build and run with Docker:

```bash
docker build --tag fcmae-torch .
```

### pre-training
Execute the pre-training script. For example:
```bash
python pretrain.py --name test_stl10 --depths 2 2 4 --dims 40 80 160
```
Then, the results are automatically saved under ```./experiments/test_stl10```.

Now, we only deel with STL10 dataset. However, we can easily extends for your desirable systems.

### fine-tuning


## License

This project is licensed under the MIT License.
