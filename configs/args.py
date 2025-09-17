import os
import yaml
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="FCMAE with ConvNeXtV2",
    )

    # Learning rates
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    
    # Encoder parameters
    parser.add_argument("--stem_stride", default=4, type=int,
                        help="kernel size of stem convolution")
    parser.add_argument("--depths", default=[2, 2, 6, 2], type=int, nargs="+",
                        help="the number of blocks in each stages")
    parser.add_argument("--dims", default=[40, 80, 160, 320], type=int, nargs="+",
                        help="hidden dimensions of each stages")
    
    # FCMAE parameters
    parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)
    parser.add_argument('--decoder_depth', type=int, default=1)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--base_lr', type=float, default=1.5e-4,
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    
    # Dataset parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--ckpt', default=None, help='resume from checkpoint')
    parser.add_argument('--num_workers', default=4, type=int)

    # output and logging
    parser.add_argument('--name', default='exp', help='experiment name')
    parser.add_argument('--save_interval', default=1, type=int)
    
    args = parser.parse_args()

    # Sanity check
    assert len(args.dims) == len(args.depths)

    # Set up output directories
    output_root  = os.path.join("./experiments", args.name)
    os.makedirs(os.path.join(output_root, "checkpoints"), exist_ok=True)
    args.output = output_root

    # Save args as YAML
    with open(os.path.join(output_root, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    
    return args


def load_config(config_path):
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