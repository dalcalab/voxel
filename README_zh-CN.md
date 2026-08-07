# Voxel

<!-- hy-mt2-i18n:start -->
[English](./readme.md) | **中文** | [日本語](./README_ja.md) | [Español](./README_es.md)
<!-- hy-mt2-i18n:end -->


这是一个用于体积（3D）医学图像分析的核心工具库，采用pytorch编写。Voxel提供了处理图像网格数据与三角形网格的方法，并充分考虑了这些数据在世界坐标系（或扫描仪坐标系）中的表示方式。该工具箱完全兼容GPU，且在开发时特别注重其在深度学习应用中的使用。更多文档即将推出！

可通过以下命令安装Voxel：

```
pip install voxel
```

如果您在工作中发现此软件包很有用，请引用以下原始开发该软件包的相关论文。
> [VoxelPrompt：一种用于端到端医学图像分析的视觉智能体](https://arxiv.org/abs/2410.08397)<br>
> Andrew Hoopes, Neel Dey, Victor Ion Butoi, John V. Guttag, Adrian V. Dalca
