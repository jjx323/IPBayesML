#!/bin/bash

## -----------------------------------------------------------------
# Backward Diffusion
# acquire the current directory current_dir
current_dir=$(pwd)

# print the directory for verification
new_dir="$current_dir/BackwardDiffusion/1D_meta_learning"
cd "$new_dir"
echo "current_dir: $PWD"

# generate data
#python generate_meta_data.py --env "simple"
#python generate_meta_data.py --env "complex"
#python generate_meta_data.py --test_true --env "simple"
#python generate_meta_data.py --test_true --env "complex"

# learn mean function and mean function with FNO
python meta_learn_mean.py --env "simple"
python meta_learn_mean.py --env "complex"
python meta_learn_FNO.py --env "simple"
python meta_learn_FNO.py --env "complex"

# Calculate MAP
python MAPSimpleCompare.py
python MAPComplexCompare.py

## -----------------------------------------------------------------
# Darcy flow problem

# print the directory for verification
new_dir="$current_dir/SteadyStateDarcyFlow/2D_meta_learn"
cd "$new_dir"
echo "current_dir: $PWD"

# generate data
#python generate_meta_data.py

# learn mean function and mean function with FNO
python meta_learn_mean.py --env "simple"
python meta_learn_mean.py --env "complex"
python meta_learn_mean_FNO.py --env "simple"
python meta_learn_mean_FNO.py --env "complex"

# Calculate MAP
python results_MAP_compare.py
python compare_truth_FNO.py --env "simple"
python compare_truth_FNO.py --env "complex"