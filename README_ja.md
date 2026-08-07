# Voxel

<!-- hy-mt2-i18n:start -->
[English](./readme.md) | [中文](./README_zh-CN.md) | **日本語** | [Español](./README_es.md)
<!-- hy-mt2-i18n:end -->


pytorchで記述された、体積型（3D）医療画像解析用のコアユーティリティです。Voxelは、画像グリッドデータや三角形メッシュを扱うためのメソッドを提供し、世界座標系（またはスキャナー座標系）におけるそれらの表現も考慮しています。このツールボックスはGPUと完全に互換性があり、特にディープラーニングアプリケーションでの利用を重視して開発されています。詳細なドキュメントもまもなく公開予定です！

Voxelは以下のコマンドでインストールできます：

```
pip install voxel
```

もしこのパッケージがご自身の業務に役立つと感じた場合は、このパッケージが元々開発された以下の論文を引用してください。
> [VoxelPrompt: 終端間医療画像解析のためのビジョンエージェント](https://arxiv.org/abs/2410.08397)<br>
> Andrew Hoopes, Neel Dey, Victor Ion Butoi, John V. Guttag, Adrian V. Dalca
