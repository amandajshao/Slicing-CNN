# Slicing Convolutional Neural Network for Crowd Video Understanding

This is the source code for ["Slicing Convolutional Neural Network for Crowd Video Understanding"](http://www.ee.cuhk.edu.hk/~jshao/papers_jshao/jshao_cvpr16_scnn.pdf). It aims at learning generic spatio-temporal features from crowd videos, especially for long-term temporal learning (i.e. `100 frames`).

## Overview

> Three-branch Slicing CNN model (i.e. xy-, xt-, and yt-branch)
>
> Crowd attribute recognition (i.e. 94 crowd-related attributes)
>
> [Project Site](http://www.ee.cuhk.edu.hk/~jshao/SCNN.html)


## Caffe
	
A fork of the well-known [Caffe](http://caffe.berkeleyvision.org/) framework with `Multi-GPU` training and `Dimension Swap` layer.

Apart from the official installation [prerequisites](http://caffe.berkeleyvision.org/installation.html), we have several other dependencies:

1. Install `openmpi` to allow multi-gpu running
2. Python packages (e.g. numpy, scipy, scikit-image, etc.)
3. Add `export PYTHONPATH="[path_python_layer]:$PYTHONPATH"` to `~/.bashrc` and restart the terminal. Here `[path_python_layer]` indicates the absolute path of the python script of `py_dim_swap_layer.py`.

Get the Caffe code

	git clone --recursive https://github.com/amandajshao/Slicing-CNN.git


## Files
- [Dataset](http://www.ee.cuhk.edu.hk/~jshao/WWWcrowd_files/www_archive.zip)

	> The dataset is introduced in CVPR 2015 which contains 10,000 crowd videos from 8,257 different crowded scenes with annotated 94 attributes.

- [LMDB Data]()

	> Release soon ...

- [CNN Initial Model](https://www.dropbox.com/s/pivm4sz5mpcp4r1/crowd_scnn_init_model.caffemodel?dl=0)

	> The initial model (VGG-16) is pre-trained on UCF-101 action dataset (single frame) and fine-tuned on WWW dataset (single frame).

- [CNN Best Model](https://www.dropbox.com/sh/qpuc7slosybj33j/AADwbKuyckmFhvkaLw95xK8oa?dl=0)

	> Three models: SCNN-xy, SCNN-xt, SCNN-yt.

- [Prototxt](https://www.dropbox.com/sh/zowetbmf9cquvmr/AABwkMFlu8I28ekBXXPrScZEa?dl=0)

	> The prototxts are corresponding to the above three models (SCNN-xy/-xt/-yt).

- Scripts

	> There are two scripts provided in our code: [model_run.sh]() and [extract_features.sh](Slicing-CNN/extract_features_xt.sh).


## Related Projects
[Deeply Learned Attributes for Crowd Scene Understanding](http://www.ee.cuhk.edu.hk/~jshao/WWWCrowdDataset.html)


## Thanks
- [Kai Kang](http://www.ee.cuhk.edu.hk/~kkang/)
- [Tong Xiao](http://www.ee.cuhk.edu.hk/~xiaotong/)
- [Yuanjun Xiong](http://personal.ie.cuhk.edu.hk/~xy012/)


## Citation

J. Shao, C. C. Loy, K. Kang, and X. Wang
Slicing Convolutional Neural Network for Crowd Video Understanding.
_Computer Vision and Pattern Recognition (CVPR), 2016_.

	@article{shao2016scnn,
	  title={Slicing Convolutional Neural Network for Crowd Video Understanding},
  	  author={Shao, Jing and Loy, Chen Change and Kang, Kai and Wang, Xiaogang},
  	  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  	  year={2016}
	}
