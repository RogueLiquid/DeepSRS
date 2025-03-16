import argparse
from pathlib import Path
from omegaconf import OmegaConf

from ResShift.sampler import ResShiftSampler

def get_parser(**parser_kwargs):
    """
    Define command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR (must be 4).")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--chop_size",
        type=int,
        default=512,
        choices=[512, 256, 64],
        help="Patch size for tiled inference."
    )
    parser.add_argument(
        "--chop_stride",
        type=int,
        default=-1,
        help="Stride between patches in tiled inference. -1 means auto-calculate."
    )
    args = parser.parse_args()
    return args

def get_configs(args):
    """
    Sets up the main config, checkpoint paths, etc.
    """
    # Adjust if your weights are still inside ResShift/weights
    ckpt_dir = Path('./ResShift/weights')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load the YAML config from your main 'configs' folder
    configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256.yaml')
    assert args.scale == 4, "We only support 4x super-resolution now!"

    # Paths to your existing model weights
    ckpt_path = ckpt_dir / 'model_300.pth'
    vqgan_path = ckpt_dir / 'autoencoder_vq_f4.pth'

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # Create output folder if it doesn't exist
    out_dir = Path(args.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto compute chop_stride if user sets -1
    if args.chop_stride < 0:
        if args.chop_size == 512:
            chop_stride = (512 - 64) * (4 // args.scale)
        elif args.chop_size == 256:
            chop_stride = (256 - 32) * (4 // args.scale)
        elif args.chop_size == 64:
            chop_stride = (64 - 16) * (4 // args.scale)
        else:
            raise ValueError("chop_size must be one of [512, 256, 64].")
    else:
        chop_stride = args.chop_stride * (4 // args.scale)

    # Adjust chop_size to actual final size
    args.chop_size *= (4 // args.scale)
    print(f"Chopping size/stride: {args.chop_size}/{chop_stride}")

    return configs, chop_stride

def main():
    """
    The main function:
      1) Parse command-line args
      2) Load configs
      3) Create sampler
      4) Run inference
    """
    # 1) Parse arguments
    args = get_parser()

    # 2) Prepare config + chop stride
    configs, chop_stride = get_configs(args)

    # 3) Instantiate the sampler
    resshift_sampler = ResShiftSampler(
        configs,
        sf=args.scale,
        chop_size=args.chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_amp=True,
        seed=args.seed,
        padding_offset=configs.model.params.get('lq_size', 64),
    )

    # 4) Run inference
    resshift_sampler.inference(
        args.in_path,
        args.out_path,
        mask_path=None,   # For plain SR, no mask is used
        bs=args.bs,
        noise_repeat=False
    )

if __name__ == "__main__":
    main()

