#!/bin/bash

nsample=512
reduced_n=256
batch_size=256

models='inception dinov2'

metrics='fd fd-infinity mmd is prdc authpct authpct_test asw ct ct_mem ct_mode fls fls_overfit'

output_dir=experiments/CIFAR10/

reference_dataset='CIFAR10:train'
test_path='CIFAR10:test'

gen_dir='data/GAN-images/CIFAR10/'

# List of generated datasets
# Metrics will be computed for each dataset of path specified here
test_datasets="CIFAR10:test \
	       CIFAR10:test \
"

for model in $models
do
	echo 'Running on model:' $model
	
	python -m dgm_eval \
	$reference_dataset \
	$test_datasets \
	--model $model  \
	--nsample $nsample -bs $batch_size \
	--output_dir $output_dir --metrics $metrics \
	--reduced_n $reduced_n --save \
	--test_path $test_path \
	# --heatmaps \

done
