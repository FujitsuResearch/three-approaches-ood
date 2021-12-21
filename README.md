#  Three approaches to facilitate DNN generalization to objects in out-of-distribution orientations and illuminations: late-stopping, tuning batch normalization and invariance loss

This repository contains the original code used in the following study.  

Akira Sakai, Taro Sunagawa, Spandan Madan, Kanata Suzuki, Takashi Katoh, Hiromichi Kobashi, Hanspeter Pfister, Pawan Sinha, Xavier Boix, and Tomotake Sasaki. [Three approaches to facilitate DNN generalization to objects in out-of-distribution orientations and illuminations: late-stopping, tuning batch normalization and invariance loss](https://arxiv.org/abs/2111.00131). arXiv preprint, arXiv:2111.00131, 2021. 

## Requirements

You can build the environment for this repository with Python3.6 and pip or docker.

### Python3.6 and pip

To install requirements:

```bash
pip install -r requirements.txt
```

### Docker

To build docker image:

```bash
./scripts/docker_build.sh
```

## Datasets

The MNIST-Positions, iLab-Orientations, CarsCG-Orientations, and MiscGoods-Illuminations datasets are based on the [MNIST](http://yann.lecun.com/exdb/mnist/), [iLab-2M ](https://bmobear.github.io/projects/viva/), [CarsCG](http://dataset.jp.fujitsu.com/data/carscg/index.html), and [DAISO-10](http://dataset.jp.fujitsu.com/data/daiso10/index.html) datasets, respectively.  

This repository contains the MNIST-Positions dataset.

To decompress the dataset, run this command:

```bash
tar -zxvf ./dataset/mnist_positions.tar.gz -C ./dataset/
```

## Training

Experiment on MNIST-Positions dataset will be run with default settings.
You can change the dataset and all parameters by modifying [run.sh](scripts/run.sh).

See [run.sh](scripts/run.sh) for detail.

### Python3.6 and pip

To train the model(s) in the paper, run this command:

```bash
./scripts/run.sh
```

### Docker

To train with docker, run this command:

```bash
# Start a TensorFlow Docker container
./scripts/docker_run.sh
# Start to train the model(s) in the container
./scripts/run.sh
```

## Outputs

A folder named ./results will be made by run.sh.
You will obtain result of experiment on MNIST-Positions medium data diversity with default setting.
Parameters in the experiment are same as that we used in our paper.


## Bibtex
```
@misc{sakai2021approaches,
      title={Three approaches to facilitate {DNN} generalization to objects in out-of-distribution orientations and illuminations: late-stopping, tuning batch normalization and invariance loss}, 
      author={Sakai, Akira  and Sunagawa, Taro  and Madan, Spandan  and Suzuki, Kanata  and Katoh, Takashi  and Kobashi, Hiromichi  and Pfister, Hanspeter  and Sinha, Pawan  and Boix, Xavier  and Sasaki, Tomotake},
      year={2021},
      howpublished={arXiv preprint, arXiv:2111.00131}, 
      note={The code is available at \url{https://github.com/FujitsuResearch/three-approaches-ood}}
}
```

## Licence
This project is under the BSD 3-Clause Clear License. See [LICENSE](LICENSE) for details.

