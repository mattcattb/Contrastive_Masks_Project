#!/bin/bash

# Define the paths
imagenet_folder="/media/mattyb/UBUNTU 22_0/datasets/imagenet_strawberries"
output_dir="/media/mattyb/UBUNTU 22_0/datasets/imagenet_strawberries/masks"
mask_type="patch"  # You can change this to the desired mask type
dir_names=("train" "unlabeled" "val")

# Iterate over the different datasets and run gen_masks.py for each
for dir_name in "${dir_names[@]}"
do
    
    dataset_dir="${imagenet_folder}/images/${dir_name}"

    #if [ ! -d "$dataset_dir"]; then
    #    echo "Directory $dataset_dir does not exist. Skipping..."
    #    continue
    #fi

    experiment_name="${dir_name}_${mask_type}"
    output="${output_dir}/${experiment_name}"
    mkdir -p "$output"

    python3 gen_masks.py --dataset_dir "${dataset_dir}" --output_dir "${output}" --mask_type "${mask_type}" --experiment_name "${experiment_name}"
done