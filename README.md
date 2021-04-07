# FSCE: Few-Shot Object Detection via Contrastive Proposal Encoding (CVPR 2021)

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ucbdrive/few-shot-object-detection.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ucbdrive/few-shot-object-detection/context:python)
This repo contains the implementation of our *state-of-the-art* fewshot object detector, described in our CVPR 2021 paper, [FSCE: Few-Shot Object Detection via Contrastive Proposal Encoding](https://arxiv.org/abs/2103.05950). FSCE is built upon the codebase [FsDet v0.1](https://github.com/ucbdrive/few-shot-object-detection/tags), which released by an ICML 2020 paper [Frustratingly Simple Few-Shot Object Detection](https://arxiv.org/abs/2003.06957).

![FSCE Figure](https://i.imgur.com/zrOSKoi.png)

### Bibtex

```
@inproceedings{FSCEv1,
 author = {Sun, Bo and Li, Banghuai and Cai, Shengcai and Yuan, Ye and Zhang, Chi},
 title = {FSCE: Few-Shot Object Detection via Contrastive Proposal Encoding},
 booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)},
 pages    = {TBD},
 month = {June},
 year = {2021}
}
```

Arxiv: https://arxiv.org/abs/2103.05950

## Contact

```
If you have any questions, please contact Bo Sun (bos [at] usc.edu) or Banghuai Li(libanghuai [at] megvii.com)
```



## Installation

FsDet is built on [Detectron2](https://github.com/facebookresearch/detectron2). But you don't need to build detectron2 seperately as this codebase is self-contained. You can follow the instructions below to install the dependencies and build `FsDet`. FSCE functionalities are implemented as `class`and `.py` scripts in FsDet which therefore requires no extra build efforts. 

**Dependencies**

* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.3 
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* Dependencies: ```pip install -r requirements.txt```
* pycocotools: ```pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'```
* [fvcore](https://github.com/facebookresearch/fvcore/): ```pip install 'git+https://github.com/facebookresearch/fvcore'``` 
* [OpenCV](https://pypi.org/project/opencv-python/), optional, needed by demo and visualization ```pip install opencv-python```
* GCC >= 4.9

**Build**

```bash
python setup.py build develop  # you might need sudo
```



Note: you may need to rebuild FsDet after reinstalling a different build of PyTorch.



## Data preparation

We adopt the same benchmarks as in FsDet, including three datasets: PASCAL VOC, COCO and LVIS. 

- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/): We use the train/val sets of PASCAL VOC 2007+2012 for training and the test set of PASCAL VOC 2007 for evaluation. We randomly split the 20 object classes into 15 base classes and 5 novel classes, and we consider 3 random splits. The splits can be found in [fsdet/data/datasets/builtin_meta.py](fsdet/data/datasets/builtin_meta.py).
- [COCO](http://cocodataset.org/): We use COCO 2014 without COCO minival for training and the 5,000 images in COCO minival for testing. We use the 20 object classes that are the same with PASCAL VOC as novel classes and use the rest as base classes.
- [LVIS](https://www.lvisdataset.org/): We treat the frequent and common classes as the base classes and the rare categories as the novel classes.

The datasets and data splits are built-in, simply make sure the directory structure agrees with [datasets/README.md](datasets/README.md) to launch the program. 

The default seed that is used to report performace in research papers can be found [here](http://dl.yf.io/fs-det/datasets/).



## Code Structure

The code structure follows Detectron2 v0.1.* and fsdet. 

- **configs**: Configuration  files (`YAML`) for train/test jobs. 
- **datasets**: Dataset files (see [Data Preparation](#data-preparation) for more details)
- **fsdet**
  - **checkpoint**: Checkpoint code.
  - **config**: Configuration code and default configurations.
  - **data**: Dataset code.
  - **engine**: Contains training and evaluation loops and hooks.
  - **evaluation**: Evaluation code for different datasets.
  - **layers**: Implementations of different layers used in models.
  - **modeling**: Code for models, including backbones, proposal networks, and prediction heads.
    - The majority of FSCE functionality are implemtended in`modeling/roi_heads/* `, `modeling/contrastive_loss.py`, and  `modeling/utils.py`
    - So one can first make sure  [FsDet v0.1](https://github.com/ucbdrive/few-shot-object-detection/tags) runs smoothly, and then refer to FSCE implementations and configurations. 
  - **solver**: Scheduler and optimizer code.
  - **structures**: Data types, such as bounding boxes and image lists.
  - **utils**: Utility functions.
- **tools**
  - **train_net.py**: Training script.
  - **test_net.py**: Testing script.
  - **ckpt_surgery.py**: Surgery on checkpoints.
  - **run_experiments.py**: Running experiments across many seeds.
  - **aggregate_seeds.py**: Aggregating results from many seeds.



## Train & Inference

### Training

We follow the eaact training procedure of FsDet and we use **random initialization** for novel weights. For a full description of training procedure, see [here](https://github.com/ucbdrive/few-shot-object-detection/blob/master/docs/TRAIN_INST.md).

#### 1. Stage 1: Training base detector.

```
python tools/train_net.py --num-gpus 8 \
        --config-file configs/PASCAL_VOC/base-training/R101_FPN_base_training_split1.yml
```

#### 2. Random initialize  weights for novel classes.

```
python tools/ckpt_surgery.py \
        --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --method randinit \
        --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1
```

This step will create a `model_surgery.pth` from` model_final.pth`. 

Don't forget the `--coco` and `--lvis`options when work on the COCO and LVIS datasets, see `ckpt_surgery.py` for all arguments details.

#### 3. Stage 2: Fine-tune for novel data.

```
python tools/train_net.py --num-gpus 8 \
        --config-file configs/PASCAL_VOC/split1/10shot_CL_IoU.yml \
        --opts MODEL.WEIGHTS WEIGHTS_PATH
```

Where `WEIGHTS_PATH` points to the `model_surgery.pth` generated from the previous step. Or you can specify it in the configuration yml. 

#### Evaluation

To evaluate the trained models, run

```angular2html
python tools/test_net.py --num-gpus 8 \
        --config-file configs/PASCAL_VOC/split1/10shot_CL_IoU.yml \
        --eval-only
```

Or you can specify `TEST.EVAL_PERIOD` in the configuation yml to evaluate during training. 



### Multiple Runs

For ease of training and evaluation over multiple runs, fsdet provided several helpful scripts in `tools/`.

You can use `tools/run_experiments.py` to do the training and evaluation. For example, to experiment on 30 seeds of the first split of PascalVOC on all shots, run

```angular2html
python tools/run_experiments.py --num-gpus 8 \
        --shots 1 2 3 5 10 --seeds 0 30 --split 1
```

After training and evaluation, you can use `tools/aggregate_seeds.py` to aggregate the results over all the seeds to obtain one set of numbers. To aggregate the 3-shot results of the above command, run

```angular2html
python tools/aggregate_seeds.py --shots 3 --seeds 30 --split 1 \
        --print --plot
```


