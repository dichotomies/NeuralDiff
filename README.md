
# NeuralDiff: Segmenting 3D objects that move in egocentric videos

## [Project Page](https://www.robots.ox.ac.uk/~vadim/neuraldiff/) | [Paper + Supplementary](https://www.robots.ox.ac.uk/~vgg/publications/2021/Tschernezki21/tschernezki21.pdf) | [Video](https://www.youtube.com/watch?v=0J98WqHMSm4)

![teaser](https://user-images.githubusercontent.com/12436822/147008441-f294a1e1-1de6-4ee1-b7c0-9872cac4f953.gif)

## About

This repository contains the official implementation of the paper *NeuralDiff: Segmenting 3D objects that move in egocentric videos* by [Vadim Tschernezki](https://github.com/dichotomies), [Diane Larlus](https://dlarlus.github.io/) and [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/). Published at 3DV21.

Given a raw video sequence taken from a freely-moving camera, we study the problem of decomposing the observed 3D scene into a static background and a dynamic foreground containing the objects that move in the video sequence. This task is reminiscent of the classic background subtraction problem, but is significantly harder because all parts of the scene, static and dynamic, generate a large apparent motion due to the camera large viewpoint change. In particular, we consider egocentric videos and further separate the dynamic component into objects and the actor that observes and moves them. We achieve this factorization by reconstructing the video via a triple-stream neural rendering network that explains the different motions based on corresponding inductive biases. We demonstrate that our method can successfully separate the different types of motion, outperforming recent neural rendering baselines at this task, and can accurately segment moving objects. We do so by assessing the method empirically on challenging videos from the EPIC-KITCHENS dataset which we augment with appropriate annotations to create a new benchmark for the task of dynamic object segmentation on unconstrained video sequences, for complex 3D environments.

## Installation

We provide an environment config file for [anaconda](https://www.anaconda.com/). You can install and activate it with the following commands:

```
conda env create -f environment.yaml
conda activate neuraldiff
```

## Dataset

The EPIC-Diff dataset can be downloaded [here](https://www.robots.ox.ac.uk/~vadim/neuraldiff/release/EPIC-Diff-annotations.tar.gz).

After downloading, move the compressed dataset to the directory of the cloned repository (e.g. `NeuralDiff`). Then, apply following commands:

```
mkdir data
mv EPIC-Diff.tar.gz data
cd data
tar -xzvf EPIC-Diff.tar.gz
```

The RGB frames are hosted separately as a subset from the [EPIC-Kitchens](https://epic-kitchens.github.io/2022) dataset. The data are available at the University of Bristol [data repository](https://doi.org/10.5523/bris.296c4vv03j7lb2ejq3874ej3vm), data.bris. Once downloaded, move the folders into the same directory as mentioned before (`data/EPIC-Diff`).

## Pretrained models

We are providing model checkpoints for all 10 scenes. You can use these to
- evaluate the models with the annotations from the EPIC-Diff benchmark
- create a summary video like at the top of this README to visualise the separation of the video into background, foreground and actor

The models can be downloaded [here](https://www.robots.ox.ac.uk/~vadim/neuraldiff/release/ckpts.tar.gz) (about 50MB in total).

Once downloaded, place `ckpts.tar.gz` into the main directory. Then execute `tar -xzvf ckpts.tar.gz`. This will create a folder `ckpts` with the pretrained models.

## Reproducing results

### Visualisations and metrics per scene

To evaluate the scene with Video ID `P01_01`, use the following command:

```
sh scripts/eval.sh rel P01_01 rel 'masks' 0 0
```

The results are saved in `results/rel`. The subfolders contain a txt file containing the mAP and PSNR scores per scene and visualisations per sample.

You can find all scene IDs in the EPIC-Diff data folder (e.g. `P01_01`, `P03_04`, ... `P21_01`).

### Average metrics over all scenes

You can calculate the average of the metrics over all scenes (Table 1 in the paper) with the following command:

```
sh scripts/eval.sh rel 0 0 'average' 0 0
```

Make sure that you have calculated the metrics per scene before proceeding with that (this command simply reads the produced metrics per scene and averages them).

### Rendering a video with separation of background, foreground and actor

To visualise the different model components of a reconstructed video (as seen on top of this page) from
1) the ground truth camera poses corresponding to the time of the video
2) and a fixed viewpoint,
use the following command:

```
sh scripts/eval.sh rel P01_01 rel 'summary' 0 0
```

This will result in a corresponding video in the folder `results/rel/P01_01/summary`.

The fixed viewpoints are pre-defined and correspond to the ones that we used in the videos provided in the supplementary material. You can adjust the viewpoints in `__init__.py` of `dataset`.

## Training

We provide scripts for the proposed model (including colour normalisation). To train a model for scene `P01_01`, use the following command.

```
sh scripts/train.sh P01_01
```

You can visualise the training with tensorboard. The logs are stored in `logs`.

## Citation

If you found our code or paper useful, then please cite our work as follows.

```bibtex
@inproceedings{tschernezki21neuraldiff,
  author     = {Vadim Tschernezki and Diane Larlus and
                Andrea Vedaldi},
  booktitle  = {Proceedings of the International Conference
                on {3D} Vision (3DV)},
  title      = {{NeuralDiff}: Segmenting {3D} objects that
                move in egocentric videos},
  year       = {2021}
}
```

## Acknowledgements

This implementation is based on [this](https://github.com/bmild/nerf) (official NeRF) and [this](https://github.com/kwea123/nerf_pl/tree/nerfw) repository (unofficial NeRF-W).

Our dataset is based on a sub-set of frames from [EPIC-Kitchens](https://epic-kitchens.github.io/2022). [COLMAP](https://colmap.github.io) was used for computing 3D information for these frames and [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/) was used for annotating them.