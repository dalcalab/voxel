---
description: Voxel is an open-source PyTorch framework for volumetric medical imaging, with GPU-accelerated, differentiable tools for MRI and CT volumes, resampling, registration, segmentation, meshes, and medical image I/O.
---

# Voxel

Voxel is a broad framework for hardware-accelerated volumetric medical imaging, centered on image and mesh data structures *and their relationship to a world coordinate system*. It is built on PyTorch, so data manipulations run on the GPU and support automatic differentiation, making it well-suited for rapid algorithm development, learning-based pipelines, or simply fast inspection of imaging data in Python.

The library provides data structures and torch-like operations spanning 3D image data and acquisition geometry, resampling, filtering, and morphology, segmentation and label maps, affine and nonlinear transformations and registration, triangular mesh processing, and broad medical imaging I/O. Install the latest release with pip.

```
pip install voxel
```

Voxel is [open source](https://github.com/dalcalab/voxel), and pull requests and bug reports are welcome.

## Citation

If voxel is useful in your work, please cite the publication it was originally developed for.

> [VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis](https://arxiv.org/abs/2410.08397)<br>
> Andrew Hoopes, Neel Dey, Victor Ion Butoi, John V. Guttag, Adrian V. Dalca
