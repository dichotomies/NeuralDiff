import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plt_to_im(f, show=False, with_alpha=False):
    # f: figure from previous plot (generated with plt.figure())
    buf = io.BytesIO()
    buf.seek(0)
    plt.savefig(buf, format="png")
    if not show:
        plt.close(f)
    im = Image.open(buf)
    # return without alpha channel (contains only 255 values)
    return np.array(im)[..., : 3 + with_alpha]


def sample_linear(X, n_samples):
    if n_samples == 0:
        n_samples = len(X)
    n_samples = min(len(X), n_samples)
    indices = (np.linspace(0, len(X) - 1, n_samples)).round().astype(np.long)
    return [X[i] for i in indices], indices


def tqdm(x, **kwargs):
    import sys

    if "ipykernel_launcher.py" in sys.argv[0]:
        # tqdm from notebook
        from tqdm.notebook import tqdm
    else:
        # otherwise default tqdm
        from tqdm import tqdm
    return tqdm(x, **kwargs)


def write_summary(results):
    """Log average precision and PSNR score for evaluation."""
    import io
    from contextlib import redirect_stdout

    with io.StringIO() as buf, redirect_stdout(buf):

        n = 0
        keys = sorted(results["metrics"].keys())

        for k in keys:
            print(k.ljust(n), end="\t")
        print()
        for k in keys:
            trail = -2 if "psnr" in k else 10
            lead = 0 if "psnr" in k else 1
            print(f'{results["metrics"][k]:.4f}'[lead:trail].ljust(n), end="\t")
        print()

        output = buf.getvalue()

    return output
