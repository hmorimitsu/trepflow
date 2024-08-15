# TrepFlow

Source code for AAAI 2025 submission 5784: **Fast Optical Flow Estimation with Temporal Re-Parameterization**.

## Installation

- It is recommended to create a new virtual environment for this installation
- Install PyTorch 2.3.1 following the instructions on the website: https://pytorch.org/get-started/previous-versions/
- Install remaining dependencies:
```bash
pip install -r requirements_trepflow.txt
```

## Usage

To run a simple demo, execute:
```bash
python demo.py
```

By default, TrepFlow (L) is used for the estimation. Other variants can be selected by providing the `--model` argument, for example:
```bash
python demo.py --model trepflow_m
```

To see a list of available model variants and all other arguments, run the following:
```bash
python demo.py -h
```

## Model variants and checkpoints

We provide all the variants and respective training checkpoints for the ablation studies from the paper.

The configurations of each variant are listed in the file [trepflow/trepflow_vars.py](trepflow/trepflow_vars.py) and the checkpoints are in the folder [ckpts](ckpts).

## Training

We train our model by adding it to the [PTLFlow platform](https://github.com/hmorimitsu/ptlflow) and running the training script. The following example uses the `trepflow_s` variant, but other models can be trained by changing to the appropriate name. TrepFlow is trained in four stages:

- Stage 1
```bash
python train.py trepflow_s --gradient_clip_val 1.0 --lr 4e-4 --wdecay 1e-4 --gamma 0.8 --train_dataset chairs --train_batch_size 10 --max_epochs 45 --random_seed 1234
```

- Stage 2
```bash
python train.py trepflow_s --pretrained path_to_stage1_ckpt --gradient_clip_val 1.0 --lr 1.25e-4 --wdecay 1e-4 --gamma 0.8 --train_dataset things --train_batch_size 6 --max_epochs 16 --random_seed 1234
```

- Stage 3
```bash
python train.py trepflow_s --pretrained path_to_stage2_ckpt --gradient_clip_val 1.0 --lr 1.25e-4 --wdecay 1e-5 --gamma 0.85 --train_dataset 200*sintel+400*kitti-2015+10*hd1k+things-train-sinteltransform --train_batch_size 6 --max_epochs 5 --random_seed 1234
```

- Stage 4
```bash
python train.py trepflow_s --pretrained path_to_stage3_ckpt --gradient_clip_val 1.0 --lr 1.25e-4 --wdecay 1e-5 --gamma 0.85 --train_dataset kitti-2015 --train_batch_size 6 --max_epochs 300 --random_seed 1234
```

## Acknowledgements

This code may contain parts from the following public repositories:

- [https://github.com/NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)
- [https://github.com/ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch)
- [https://github.com/hmorimitsu/ptlflow](https://github.com/hmorimitsu/ptlflow)
- [https://github.com/princeton-vl/RAFT/](https://github.com/princeton-vl/RAFT/)
- [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)