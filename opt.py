import argparse

import git


def get_opts(vid=None, root="data/EPIC-Diff"):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root", type=str, default=root, help="Root directory of dataset."
    )
    parser.add_argument(
        "--N_emb_xyz", type=int, default=10, help="Number of xyz embedding frequencies."
    )
    parser.add_argument(
        "--N_emb_dir",
        type=int,
        default=4,
        help="Number of direction embedding frequencies.",
    )
    parser.add_argument(
        "--N_samples", type=int, default=64, help="Number of coarse samples."
    )
    parser.add_argument(
        "--N_importance",
        type=int,
        default=64,
        help="Number of additional fine samples.",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="Factor to perturb depth sampling points.",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=1.0,
        help="Std dev of noise added to regularize sigma.",
    )
    parser.add_argument(
        "--N_vocab",
        type=int,
        default=1000,
        help="Number of frames (max. 1000 for our dataset).",
    )
    parser.add_argument(
        "--N_a", type=int, default=48, help="Embedding size for appearance encoding."
    )
    parser.add_argument(
        "--N_tau",
        type=int,
        default=17,
        help="Embedding size for transient encoding.",
    )
    parser.add_argument(
        "--beta_min",
        type=float,
        default=0.03,
        help="Minimum color variance for loss.",
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size.")
    parser.add_argument(
        "--chunk",
        type=int,
        default=32 * 1024,
        help="Chunk size to split the input to avoid reduce memory footprint.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of gpus.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Pretrained checkpoint path to load.",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay.")
    parser.add_argument("--exp_name", type=str, default="exp", help="Experiment name.")
    parser.add_argument(
        "--refresh_every",
        type=int,
        default=1,
        help="print the progress bar every X steps",
    )
    parser.add_argument("-f", type=str, default="", help="For Jupyter.")
    parser.add_argument(
        "--lowpass_K",
        type=int,
        default=21,
        help="K for low rank expansion of transient encoding.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="Fraction of train dataset to use per epoch. For debugging.",
    )
    parser.add_argument(
        "--model_width", type=int, default=256, help="Width of model (units per layer)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloaders."
    )
    parser.add_argument("--vid", type=str, default=vid, help="Video ID of dataset.")
    parser.add_argument(
        "--deterministic", default=True, action="store_true", help="Reproducibility."
    )
    parser.add_argument(
        "--inference",
        default=False,
        action="store_true",
        help="For compatibility with evaluation script.",
    )

    hparams, unknown = parser.parse_known_args()
    if unknown:
        # for compabitibility with evaluation script.
        if "--is_eval_script" not in unknown:
            print("--- unrecognised arguments ---")
            print(unknown)
            exit()
    hparams.git_train = git.Repo(search_parent_directories=True).head.object.hexsha
    # placeholders for eval script
    hparams.git_eval = ""
    hparams.ckpt_path_eval = ""

    return hparams


if __name__ == "__main__":
    hparams = get_opts("example")

    print("Argparse options:")
    for k, v in hparams.__dict__.items():
        print(f"{k}: {v}")
