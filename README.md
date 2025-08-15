# Euclidean neural networks on Apple Silicon

## Introduction
This repository provides a lightweight port of the e3nn library. The primary objective is to enable UMA and other MLIPs on Apple chips. Only the essential components have been ported to achieve compatibility and basic functionality.

The aim of this library is to help the development of [E(3)](https://en.wikipedia.org/wiki/Euclidean_group) equivariant neural networks.
It contains fundamental mathematical operations such as [tensor products](https://docs.e3nn.org/en/stable/api/o3/o3_tp.html) and [spherical harmonics](https://docs.e3nn.org/en/stable/api/o3/o3_sh.html).

![](https://user-images.githubusercontent.com/333780/79220728-dbe82c00-7e54-11ea-82c7-b3acbd9b2246.gif)

## Citing

If you use e3nn in your research, please cite the following papers:

### Euclidean Neural Networks:

- N. Thomas et al., "Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds" (2018). [arXiv:1802.08219](https://arxiv.org/abs/1802.08219)
- M. Weiler et al., "3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data" (2018). [arXiv:1807.02547](https://arxiv.org/abs/1807.02547)
- R. Kondor et al., "Clebsch-Gordan Nets: a Fully Fourier Space Spherical Convolutional Neural Network" (2018). [arXiv:1806.09231](https://arxiv.org/abs/1806.09231)

### e3nn:

- M. Geiger and T. Smidt, "e3nn: Euclidean Neural Networks" (2022). [arXiv:2207.09453](https://arxiv.org/abs/2207.09453)
- M. Geiger et al., "Euclidean neural networks: e3nn" (2022). [Zenodo](https://doi.org/10.5281/zenodo.6459381)
