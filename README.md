<h1 align="center">A LoD of Gaussians: Out-of-Core Training and Rendering for Seamless
Ultra-Large Scene Reconstruction</h1>

<p align="center">
  <a href="https://felixwindisch.github.io/ALoDOfGaussians/">
    <img src="https://img.shields.io/badge/Project-Page-darkblue" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2507.01110">
    <img src="https://img.shields.io/badge/arXiv-2603.24725-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://cloud.tugraz.at/index.php/s/tRz85cJsRQGJX4q">
    <img src="https://img.shields.io/badge/Data-Uni10k-darkorange" alt="Point Clouds">
  </a>
  <a href="https://youtu.be/5mRpZGSqoyg">
    <img src="https://img.shields.io/badge/Video-YouTube-red" alt="Video">
  </a>
</p>

<h3 align="center">SIGGRAPH 2026</h3>

<h4 align="center">
    <a href="https://felixwindisch.github.io/">Felix Windisch</a><sup>1</sup> ·
    <a href="https://derthomy.github.io/">Thomas Köhler</a><sup>1</sup> ·
    <a href="https://r4dl.github.io/">Lukas Radl</a><sup>1</sup> ·
    <a href="https://mattiadurso.github.io/">Mattia D'Urso</a><sup>1</sup> ·
    <a href="https://steimich96.github.io/">Michael Steiner</a><sup>1</sup> ·
    <a href="https://schmalstieg.github.io/">Dieter Schmalstieg</a><sup>1</sup> ·
    <a href="https://www.markussteinberger.net/">Markus Steinberger</a><sup>1,2</sup>
</h4>

  <div align="center">
    <p>
      <sup>1</sup> Graz University of Technology 🇦🇹<br>
      <sup>2</sup> Huawei Technologies 🇦🇹
    </p>
  </div>

## Overview
**A LoD of Gaussians** enables seamless ultra-large 3DGS training and rendering on consumer GPUs through a combination of out-of-core streaming and level of detail.
This repository contains the official authors' implementation associated with the paper "A LoD of Gaussians: Unified Training and Rendering for Ultra-Large-Scale Reconstruction with External Memory". 
## Setup

Make sure to clone the repository using `--recursive`:
```
git clone -b Refactor https://github.com/FelixWindisch/hierarchical-LOD-gaussians.git --recursive
cd hierarchical-LOD-gaussians
```

Setting up the conda environment:
```
conda create -n LoDOfFGaussians
conda activate LoDOfFGaussians
conda install python=3.10
conda install -c nvidia cuda-toolkit=12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```
Install the last 5 dependencies with --no-build-isolation if you get errors.

### Compiling hierarchy generator and merger
These files were adapted from Hierarchical 3DGS and can be built as follows:
```
cd submodules/gaussianhierarchy
cmake . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --config Release
cd ../..
```
## Running the method

#### Dataset 
Prepare your dataset in the standard 3DGS format:
```
root/
├─ sparse/
│  ├─ 0/
│  │  ├─ cameras.bin
│  │  ├─ images.txt
│  │  ├─ points3D.txt
├─ images/
├─ masks/
├─ depths/
```
If depth images or masks are used, place them in root/depths and root/masks respectively.
To start training, execute:
```
python train.py --project_dir root --config default.json --skip_if_exists
```
Check out the (Hyperparamter Guide)[Docs/Hyperparameters.md] to create your own configuration file tailored for your scene.

Training runs in 2 steps: Coarse Optimization (Standard 3DGS, sparse point cloud) and Fine Optimization (Out of Core and LoD, with densification). 

After finishing both steps, a _out.dhier file will be written to ```root/outputs```, which can be rendered and evaluated:
```
python eval_hierarchy.py --hierarchy_path /path/to/result_out.dhier -s root/  --config default.json
python hierarchy_viewer.py --hierarchy_path /path/to/result_out.dhier -s root/  --config default.json
```
```eval_hierarchy``` will render all images in the test set (use the llffhold in your config parameter to designate every nth image for testing) and output quality metrics.
```hierarchy_viewer``` allows interactive viewing of the results. This can be done using the networked inria viewer, but we strongly recommend installing SplatViz (https://github.com/Florian-Barthel/splatviz) and running it with ```python run_main.py --mode=attach``` while ```hierarchy_viewer``` is running.

### Disclaimer
Note that this code release version relies on the gsplat rasterizer and will thus be more memory-efficient than reported in the paper.

