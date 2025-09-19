# FCMAE-torch

A minimal, flexible PyTorch implementation of the Fully Convolutional Masked AutoEncoder (FCMAE), designed to operate **without** MinkowskiEngine and natively support non-square images.

## Features

- 🚀 **Pure PyTorch**: No MinkowskiEngine dependency
- 🖼️ **Non-square Image Support**: Works seamlessly with images of any aspect ratio
- 🛠️ **Readable & Extensible**: Clean codebase for easy customization

## Getting Started

### Installation

Clone the repository and set up the environment:

```bash
bash docker_run.sh
```

This command will build and launch a Docker container with all necessary dependencies pre-installed.

### Pre-training

Run the pre-training script. For example:

```bash
python pretrain.py --name test_stl10
```

Results will be saved to `./experiments/test_stl10`.

Currently, the codebase is configured for the STL10 dataset, but it can be easily adapted for other datasets.

To evaluate on the test set and visualize results, use:

```bash
python test_fcmae.py --name test_stl10
```

Inference visualizations are saved as `./experiments/test_stl10/test_viz.mp4`.

### Fine-tuning

After pre-training, fine-tune your model with:

```bash
python finetune.py --pre test_stl10 --name test_stl10_ft
```

- Use `--pre` to specify the pre-trained experiment name.
- Fine-tuned results are saved under `./experiments/test_stl10_ft`.

For baseline comparison, you can run supervised learning from scratch by adding the `--without_pre` flag.

## Results

### Pre-training

in preparation

### Fine-tuning vs. Supervised Learning

in preparation

## References
- [ConvNeXtV2](https://arxiv.org/abs/2301.00808)
- [STL10](https://cs.stanford.edu/~acoates/stl10/)

## License

This project is licensed under the MIT License.
