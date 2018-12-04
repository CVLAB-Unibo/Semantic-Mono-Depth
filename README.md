# Semantic-Mono-Depth

This repository contains the source code of Semantic-Mono-Depth, proposed in the paper "Geometry meets semantics for semi-supervised monocular depth estimation", ACCV 2018.
If you use this code in your projects, please cite our paper:

```
@inproceedings{3net18,
  title     = {Geometry meets semantics for semi-supervised monocular depth estimation},
  author    = {Pierluigi Zama Ramirez and
               Matteo Poggi and
               Fabio Tosi and
               Stefano Mattoccia and
               Luigi Di Stefano},
  booktitle = {14th Asian Conference on Computer Vision (ACCV)},
  year = {2018}
}
```

For more details:
[arXiv](https://arxiv.org/abs/1810.04093)

## Requirements

* `Tensorflow 1.5 or higher` (recomended) 
* `python packages` such as opencv, matplotlib

## Download pretrain models
Checkpoints can be downloaded from [here](https://drive.google.com/open?id=1n4qPzso_uyodgevi3w0qCXduTsPXqlub)

## Inference and evaluation
```
python monodepth_main.py --dataset kitti --mode test --data_path $DATA_PATH --output_dir $OUTPUT_DIR --filename ./utils/filenames/kitti_semantic_stereo_2015_test_split.txt --task depth --checkpoint_path $checkpoint_path --encoder $ENCODER

python ./utils/evaluate_kitti.py --split kitti_test --predicted_disp_path $OUTPUT_DIR/disparities_pp.npy --gt_path $DATA_PATH 
```

DATA_PATH=`path_to_dataset`
OUTPUT_DIR=`path_to_output_folder`
ENCODER=`vgg` or `resnet`