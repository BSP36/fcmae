import os
import yaml
import argparse

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training a Fully Convolutional Masked Autoencoder (FCMAE) with ConvNeXtV2.

    Returns:
        argparse.Namespace: Configuration parameters as a namespace.
    """
    parser = argparse.ArgumentParser(description="FCMAE with ConvNeXtV2")
    # Training parameters
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size for training.')
    parser.add_argument('--epochs', default=300, type=int,
                        help='Total number of training epochs.')
    parser.add_argument('--warmup_epochs', type=int, default=4,
                        help='Number of warm-up epochs for learning rate scheduling.')
    
    # Encoder parameters
    parser.add_argument("--stem_stride", default=2, type=int,
                        help="Stride of the stem convolution layer.")
    parser.add_argument("--depths", default=[2, 2, 6, 2], type=int, nargs="+",
                        help="Number of blocks in each encoder stage.")
    parser.add_argument("--dims", default=[40, 80, 160, 320], type=int, nargs="+",
                        help="Hidden dimensions for each encoder stage.")

    # FCMAE-specific parameters
    parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use per-patch normalized pixel values as reconstruction targets.')
    parser.set_defaults(norm_pix_loss=True)
    parser.add_argument('--decoder_depth', type=int, default=1,
                        help='Number of ConvNeXtV2 blocks in the decoder.')
    parser.add_argument('--decoder_embed_dim', type=int, default=512,
                        help='Embedding dimension for the decoder.')

    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--base_lr', type=float, default=1.5e-4,
                        help='Base learning rate')
    
    # Dataset and device parameters
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training and evaluation (e.g., "cuda" or "cpu").')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of data loading workers.')

    # Output and logging
    parser.add_argument('--name', default='exp',
                        help='Name of the experiment.')
    parser.add_argument('--save_interval', default=10, type=int,
                        help='Interval (in epochs) for saving checkpoints.')

    
    args = parser.parse_args()
    args = prepare_dirs(args)
    
    return args


def parse_args_ft() -> argparse.Namespace:
    """Parse command-line arguments for fine-tuning a Fully Convolutional Masked Autoencoder (FCMAE).

    Returns:
        argparse.Namespace: Configuration parameters as a namespace.
    """
    # Additional arguments for fine-tuning
    parser = argparse.ArgumentParser(description='Fine-tuning script for FCMAE')

    parser.add_argument('--pre', type=str,
                        help='Name of the experiment containing the pretrained model.')
    parser.add_argument('--name', type=str,
                        help='Name of the current fine-tuning experiment.')
    parser.add_argument('--without_pre', action='store_true',
                        help='Train the model from scratch without using a pretrained checkpoint '
                             '(intended for comparative experiments).')
    parser.add_argument('--base_lr', type=float, default=1.5e-4,
                        help='Base learning rate.')
    parser.add_argument('--bb_mult', type=float, default=1.0,
                        help='Scaling factor for the backbone learning rate. '
                             'Backbone LR = base_lr * bb_mult.')

    args_ft = parser.parse_args()

    # Load configuration from the pretrained experiment
    config_path = os.path.join("./experiments", args_ft.pre, "config.yaml")
    args = load_config(config_path)
    args.ckpt = os.path.join(args.output, "checkpoints", "fcmae_best.pth")
    if args_ft.without_pre:
        print("Training without a pretrained checkpoint.")
        args.ckpt = None

    
    # Update configuration with fine-tuning arguments
    for key, value in vars(args_ft).items():
        setattr(args, key, value)

    # Prepare directories for outputs
    args = prepare_dirs(args)

    return args


def prepare_dirs(args: argparse.Namespace) -> argparse.Namespace:
    """Prepare output directories and save the configuration.

    This function performs the following:
    - Validates that the encoder dimensions and depths match.
    - Creates necessary output directories for the experiment.
    - Saves the current configuration as a YAML file.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        argparse.Namespace: Updated arguments with output paths.
    """
    # Validate encoder configuration
    assert len(args.dims) == len(args.depths), \
        "Mismatch: 'dims' and 'depths' must have the same length."

    # Set up the main output directory
    args.output = os.path.join("./experiments", args.name)

    # Create subdirectories for checkpoints and results
    for subdir in ["checkpoints", "results"]:
        os.makedirs(os.path.join(args.output, subdir), exist_ok=True)

    # Save configuration as YAML
    config_path = os.path.join(args.output, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(vars(args), f)

    return args
    

def load_config(config_path: str):
    """
    Load configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        argparse.Namespace: Configuration parameters as a namespace.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)