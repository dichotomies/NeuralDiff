import os

from PIL import Image
from tqdm import tqdm

from . import convert_file_format


def resize_image(PIL_image, h, w):
    if (w, h) != PIL_image.size:
        im = PIL_image.resize((int(w), int(h)), Image.LANCZOS)
    else:
        im = PIL_image
    return im


def load_image(path):
    im = Image.open(path).convert("RGB")
    return im


def save_frames(dir_src, dir_dst, meta, format_src=None, format_trg=None):
    for k in tqdm(meta["ids_all"]):
        name_src = convert_file_format(meta["images"][k], format_src)
        path_src = os.path.join(dir_src, name_src)
        name_trg = convert_file_format(name_src, format_trg)
        path_trg = os.path.join(dir_dst, name_trg)
        frame = resize_image(load_image(path_src), meta["image_h"], meta["image_w"])
        frame.save(path_trg)


def save_annotations(root, dataset, maskloader):
    """NOTE: not used for now."""
    root = os.path.join(root, dataset.vid, "annotations")
    os.makedirs(root)

    for k in dataset.img_ids_test:
        mask_orig = maskloader[k, 0, 1][0]
        mask = PIL.Image.fromarray(mask_orig)
        sample = dataset[k]
        im_path = sample["im_path"]
        path = os.path.join(root, im_path.replace("jpg", "bmp"))
        mask.save(path, format="bmp")
        mask = np.array(PIL.Image.open(path))
        assert (mask == mask_orig).all()
