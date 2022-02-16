# Look at What I’m Doing: Self-Supervised Spatial Grounding of Narrations in Instructional Videos

![alt text](motivational.png)

This repository contains a PyTorch implementation of the paper [Look at What I’m Doing: Self-Supervised Spatial Grounding of Narrations in Instructional Videos](https://proceedings.neurips.cc/paper/2021/file/792dd774336314c3c27a04bb260cf2cf-Paper.pdf) accepted at NeurIPS 2021 (spotlight). If you find this implementation or the paper helpful, please consider citing:

    @InProceedings{tanCOMMA2021,
         author={Reuben Tan and Bryan A. Plummer and Kate Saenko and Hailin Jin and Bryan Russell},
         title={Look at What I’m Doing: Self-Supervised Spatial Grounding of Narrations in Instructional Videos},
         booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
         year={2021} }

# Dependencies

1. Python 3.6
2. Pytorch version 1.7.0
3. Ffmpeg
4. Open-CV


# Project Code Files
The code is currently getting cleaned up and tested. It will be released very soon! Thank you for your patience.

# Download YouCook2-Interactions Dataset

Please go to this [link](https://drive.google.com/file/d/123HrerGvNZZO9GosvhccaqMw6lAl5L0u/view?usp=sharing) to download the YouCook2-Interactions evaluation dataset and unzip it. The output folder has the following files:

1. final_dataset_segments.pkl - this file contains all the video segments that are used for evaluation. Each segment is represented by a tuple where its elements are the video name and start and end times in seconds.
2. final_dataset_annotations.pkl - this file contains the frame-level bounding box annotations for the video segments.

If you are interested in visualizing the YouCook2-Interactions dataset, you can do so by running the following command:

`python plot_local_annotations.py --video_dir {directory where YouCook2 videos are stored} --annotations_path {path to final_dataset_annotations.pkl} --segments_path {path to final_dataset_segments.pkl} --output_dir {directory to store annotated frames}`

# Preprocess videos into bytestream files (optional)

# Training code

# Evaluation code

Before starting the evaluation, please download the original YouCook2 train and validation annotations [here](http://youcook2.eecs.umich.edu/static/YouCookII/youcookii_annotations_trainval.tar.gz). 

To run the evaluation code, please run the following command:

`python -W ignore eval_youcook_interactions_localization.py --eval_video_root {directory containing YouCook2 videos}  --youcook2_annotations_path {path to json file containing YouCook2 annotations} --interactions_annotations_path {path to YouCook2-Interaction annotations file} --interactions_segments_path {path to YouCook2-Interaction segments file} ----checkpoint_eval {path to trained model weights}`

# Code credit