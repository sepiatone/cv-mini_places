#### MiniPlaces Challenge

This is a submission for the [MiniPlaces Challege](http://6.869.csail.mit.edu/fa17/miniplaces.html), an assignment for the course Computer Vision (MIT 6.869, Fall 2017). 

The goal of this challenge is to identify the scene category depicted in a photograph. The [dataset](http://miniplaces.csail.mit.edu/data/data.tar.gz) for this task comes from a subset of the [Places2](http://places2.csail.mit.edu/) dataset. Specifically, the mini challenge data for this course will be a subsample of the above data, consisting of 100,000 images for training, 10,000 images for validation and 10,000 images for testing coming from 100 scene categories. The images will be resized to 128\*128 to make the data more manageable. Further, while the end goal is scene recognition, a subset of the data will contain object labels that might be helpful to build better models.

An overview and rationale for the MiniPlaces Challenge as a pedagogical tool to learn deep learning techniques for Computer Vision may be found [here](https://github.com/CSAILVision/miniplaces).

Example command for training

	python train.py \
	 --model 'resnet18' \
	 --num_epochs 10 \
	 --data_path 'data/' \
	 --result_path 'data/' \
