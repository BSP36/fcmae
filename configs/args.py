import os
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="FCMAE with ConvNeXtV2")
    # Learning rates
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=4, help='epochs to warmup LR')
    
    # Encoder parameters
    parser.add_argument("--stem_stride", default=2, type=int,
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
                        help='weight decay')
    parser.add_argument('--base_lr', type=float, default=1.5e-4,
                        help='base learning rate')
    
    # Dataset parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=8, type=int)

    # output and logging
    parser.add_argument('--name', default='exp', help='experiment name')
    parser.add_argument('--save_interval', default=10, type=int)
    
    args = parser.parse_args()
    args = prepare_dirs(args)
    
    return args



def parse_args_ft():
    # additional args for fine-tuning
    parser = argparse.ArgumentParser('FCMAE fine-tuning script', add_help=False)
    parser.add_argument('--pre', type=str, help='experiment name of pretinaed model')
    parser.add_argument('--name', type=str, help='experiment name')
    parser.add_argument('--without_pre', action='store_true',
                        help='Train model without a pretrained ckpt (only for comparative experiment)')
    parser.add_argument('--base_lr', type=float, default=1.5e-4,
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--bb_mult', type=float, default=1.0,
                        help='The backbone is trained with base_lr * bb_mult')
    args_ft = parser.parse_args()

    # Load config
    config_path = os.path.join("./experiments", args_ft.pre, "config.yaml")
    args = load_config(config_path)
    args.ckpt = os.path.join(args.output, "checkpoints", "fcmae_best.pth")
    if args_ft.without_pre:
        print("Train without pretrained checkpoint")
        args.ckpt = None

    # Update
    for key, value in vars(args_ft).items():
        setattr(args, key, value)

    # Prepare
    args = prepare_dirs(args)

    return args


def prepare_dirs(args):
    # Sanity check
    assert len(args.dims) == len(args.depths)

    # Set up output directories
    args.output  = os.path.join("./experiments", args.name)
    for key in ["checkpoints", "results"]:
        os.makedirs(os.path.join(args.output, key), exist_ok=True)

    # Save args as YAML
    with open(os.path.join(args.output, "config.yaml"), "w") as f:
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