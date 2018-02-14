# voxelmorph

Initial version of voxelmorph code.


## Notes
- Code is written in python 2.7. A 3.5 version is on the way!

- We are currently cleaning up our code for general use. There are several hard-coded elements related
to data preprocessing and format. You will likely need to rewrite some of the data loading code in 
'datagenerator.py' for your own datasets.

- We provide the atlas used in our papers at data/atlas_norm.npz.

## Papers
**An Unsupervised Learning Model for Deformable Medical Image Registration**  
Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca  
[eprint arXiv:1802.02604](https://arxiv.org/abs/1802.02604)

## Instructions

### Training:

1. Change base_data_dir in train.py to the location of your image files.
2. Run train.py [model_name] [gpu-id] 

### Testing (Dice scores):
Put test filenames in data/test_examples.txt, and anatomical labels in data/test_labels.mat.
1. Run test.py [model_name] [gpu-id] [iter-num]
