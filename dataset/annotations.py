import os

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image


def blend_mask(im, mask, colour, alpha, show_im=False):
    """Blend an image with a mask (colourised via `colour` and `alpha`)."""
    im = im.copy().astype(np.float) / 255
    for ch, rgb_v in zip([0, 1, 2], colour):
        im[:, :, ch][mask == 1] = im[:, :, ch][mask == 1] * (1 - alpha) + rgb_v * alpha
    if show_im:
        plt.imshow(im)
        plt.axis("off")
        plt.show()
    return im


class MaskLoader:
    """Loads masks for a dataset initialised with a video ID."""

    def __init__(self, dataset, is_debug=False):
        self.frames_dir = os.path.join(dataset.root, "frames")
        self.annotations_dir = os.path.join(dataset.root, "annotations")
        self.image_paths = dataset.image_paths

        self.mask_colour = [1, 0, 0]
        self.mask_alpha = 0.5

        self.is_debug = is_debug

        print(f"ID of loaded scene: {dataset.vid}.")
        print(f"Number of annotations: {len(os.listdir(self.annotations_dir))}.")

    def __getitem__(self, sample_id):
        image_id, image_ext = self.image_paths[sample_id].split(".")

        im = plt.imread(os.path.join(self.frames_dir, image_id + "." + image_ext))
        mask = np.array(
            PIL.Image.open(
                os.path.join(self.annotations_dir, image_id + "." + image_ext)
            )
        )

        if self.is_debug:
            blend_mask(im, mask, self.mask_colour, self.mask_alpha, True)

        return mask, im
