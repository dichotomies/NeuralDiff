import json
import os
from copy import deepcopy
from os.path import join

import numpy as np
import torch
from . import convert_file_format

from .colmap_utils import (read_cameras_binary, read_images_binary,
                           read_points3d_binary)


def compare_meta(meta1, meta2):
    assert len(meta1) == len(meta2)
    for k in meta1:
        if any([x in k for x in ["ids", "image"]]):
            assert meta1[k] == meta2[k]
        elif k in ["poses", "nears", "fars"]:
            assert (
                np.array(list(meta1["poses"].values()))
                == np.array(list(meta2["poses"].values()))
            ).all()
        else:
            assert (meta1[k] == meta2[k]).all()


def load_colmap(path_model):
    colmap = {}
    colmap["images"] = read_images_binary(join(path_model, "images.bin"))
    colmap["cameras"] = read_cameras_binary(join(path_model, "cameras.bin"))
    colmap["pts3d"] = read_points3d_binary(join(path_model, "points3D.bin"))
    return colmap


def split_ids(ids, nth_image=8):

    if nth_image != 0:
        ids_te = ids[::nth_image][::2]
        ids_vl = ids[::nth_image][1::2]
    else:
        ids_te = []
        ids_vl = []
    ids_tr = list(sorted(set(ids).difference(set(ids_vl).union(ids_te))))

    return ids_tr, ids_vl, ids_te


def calculate_intrinsics(colmap_camera, downscale=1):
    K = np.zeros((3, 3), dtype=np.float32)
    s = 1 / downscale
    K[0, 0] = colmap_camera.params[0] * s  # fx
    K[1, 1] = colmap_camera.params[0] * s  # fy
    K[0, 2] = colmap_camera.params[1] * s  # cx
    K[1, 2] = colmap_camera.params[2] * s  # cy
    K[2, 2] = 1
    return K


def reduce_ids(meta):
    """Reduce large filename IDs to small ones for compatibility with time embedding."""
    min_id = min(meta['ids_all'])
    fn_reduce = lambda x, min_id: (x - min_id) // 50
    assert len(set([fn_reduce(x, min_id) for x in meta['ids_all']])) == len(set(meta['ids_all']))
    for k in meta:
        if 'ids' in k:
            meta[k] = [fn_reduce(x, min_id) for x in meta[k]]
        elif k in ['nears', 'fars', 'poses', 'images']:
            meta[k] = {fn_reduce(k, min_id): v for k, v in meta[k].items()}


def calc_meta(colmap, image_downscale=1, with_cuda=True, split_nth=8):

    # e.g., `IMG_0000000200.bmp` to 200
    fn2int = lambda fn: int(os.path.splitext(fn.split("_")[1])[0])

    # 1: load cameras and sort COLMAP indices
    # (COLMAP indices are not necessariliy sorted)

    colmap2fn = dict(
        sorted(
            [(k, colmap["images"][k].name) for k in colmap["images"]],
            key=lambda x: x[0],
        )
    )

    assert list(colmap2fn.keys()) == sorted(list(colmap2fn.keys()))
    fn2colmap = {v: k for k, v in colmap2fn.items()}
    colmap2sortedfn = dict(zip(colmap2fn, sorted(colmap2fn.values())))
    colmap2sortedcolmap = {k: fn2colmap[v] for k, v in colmap2sortedfn.items()}
    colmap2sortedframeid = {k: fn2int(fn) for k, fn in colmap2sortedfn.items()}
    assert list(colmap2sortedframeid.values()) == sorted(
        list(colmap2sortedframeid.values())
    )

    colmap["images"] = {
        fn2int(colmap["images"][colmap2sortedcolmap[i]].name): colmap["images"][
            colmap2sortedcolmap[i]
        ]
        for i in colmap["images"]
    }

    meta = {"ids_all": []}

    meta["images"] = {}  # {id: filename}
    for k, v in colmap["images"].items():
        filename = v.name
        meta["images"][k] = filename
        meta["ids_all"] += [k]

    # 2: read and rescale camera intrinsics

    assert len(colmap["cameras"]) == 1
    colmap_camera = list(colmap["cameras"].values())[0]
    meta["intrinsics"] = calculate_intrinsics(colmap_camera, downscale=image_downscale)
    meta['camera'] = {}
    for k in ['model', 'width', 'height', 'params']:
        if k == 'params':
            meta['camera'][k] = getattr(colmap_camera, k).tolist()
        else:
            meta['camera'][k] = getattr(colmap_camera, k)

    meta["image_w"] = colmap_camera.params[1] * 2 * (1 / image_downscale)
    meta["image_h"] = colmap_camera.params[2] * 2 * (1 / image_downscale)

    # 3: read w2c and initialise c2w (poses) from w2c

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.0]).reshape(1, 4)
    for id_ in meta["ids_all"]:
        im = colmap["images"][id_]
        R = im.qvec2rotmat()
        t = im.tvec.reshape(3, 1)
        w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
    w2c_mats = np.stack(w2c_mats, 0)  # (N_images, 4, 4)
    poses = np.linalg.inv(w2c_mats)[:, :3]  # (N_images, 3, 4)
    # poses has rotation in form "right down front", change to "right up back"
    poses[..., 1:3] *= -1

    xyz_world = np.array([colmap["pts3d"][i].xyz for i in colmap["pts3d"]])
    xyz_world_h = np.concatenate([xyz_world, np.ones((len(xyz_world), 1))], -1)

    # 4: near and far bounds for each image

    from tqdm.notebook import tqdm

    # speed up if 1000s of images to be used
    if with_cuda:
        to = lambda x: x.cuda()
    else:
        to = lambda x: x

    w2c_mats_pt = torch.from_numpy(w2c_mats)
    xyz_world_h_pt = to(torch.from_numpy(xyz_world_h))

    meta["nears"], meta["fars"] = {}, {}
    n_ids = len(meta["ids_all"])
    for i, id_ in tqdm(enumerate(meta["ids_all"]), total=n_ids, disable=n_ids < 1000):
        xyz_cam_i = (xyz_world_h_pt @ (to(w2c_mats_pt[i].T)))[:, :3]
        xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2] > 0]
        meta["nears"][id_] = torch.quantile((xyz_cam_i[:, 2]), 0.1 / 100).item()
        meta["fars"][id_] = torch.quantile((xyz_cam_i[:, 2]), 99.9 / 100).item()

    meta["poses"] = {id_: poses[i] for i, id_ in enumerate(meta["ids_all"])}
    meta["ids_train"], meta["ids_val"], meta["ids_test"] = split_ids(
        meta["ids_all"], nth_image=split_nth
    )

    # reduce_ids(meta)

    return meta


def update_format(meta, image_format=None):
    if format is not None:
        for k in meta['images']:
            meta['images'][k] = convert_file_format(meta['images'][k], image_format)


def load_meta(directory, name="meta.json"):
    path = os.path.join(directory, name)
    with open(path, "r") as fp:
        meta = json.load(fp)
    for k in ["nears", "fars", "images"]:
        meta[k] = {int(i): meta[k][i] for i in meta[k]}
    meta["poses"] = {int(i): np.array(meta["poses"][i]) for i in meta["poses"]}
    meta["intrinsics"] = np.array(meta["intrinsics"])
    return meta


def save_meta(directory, meta, name="meta.json", deepcopy=False):
    tolist = lambda x: x if type(x) is list else x.tolist()
    path = os.path.join(directory, name)
    if deepcopy:
        meta = deepcopy(meta)

    meta["poses"] = {k: tolist(meta["poses"][k]) for k in meta["poses"]}
    meta["intrinsics"] = tolist(meta["intrinsics"])
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
