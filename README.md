# Efficient Track Anything
[[`📕Project`](https://yformer.github.io/efficient-track-anything/)][[`🤗Gradio Demo`](https://55e434edafdc3f1c34.gradio.live)][[`📕Paper`](https://arxiv.org/pdf/2411.18933)]

![Efficient Track Anything Speed](figs/examples/latency.png)

The **Efficient Track Anything Model(EfficientTAM)** takes a vanilla lightweight ViT image encoder. An efficient memory cross-attention is proposed to further improve the efficiency. Our EfficientTAMs are trained on SA-1B (image) and SA-V (video) datasets. EfficientTAM achieves comparable performance with SAM 2 with improved efficiency. Our EfficientTAM can run **>10 frames per second** with reasonable video segmentation performance on **iPhone 15**. Try our demo with a family of EfficientTAMs at [[`🤗Gradio Demo`](https://55e434edafdc3f1c34.gradio.live)].

![Efficient Track Anything design](figs/examples/overview.png)

## News
[Dec.2 2024] We release the codebase of Efficient Track Anything.

## Online Demo & Examples
Online demo and examples can be found in the [project page](https://yformer.github.io/efficient-track-anything/).

## EfficientTAM Video Segmentation Examples
  |   |   |
:-------------------------:|:-------------------------:
SAM 2 | ![SAM2](figs/examples/sam2_video_segmentation.png)
EfficientTAM |  ![EfficientTAM](figs/examples/efficienttam_video_segmentation.png)

## EfficientTAM Image Segmentation Examples
Input Image, SAM, EficientSAM, SAM 2, EfficientTAM
  |   |   |
:-------------------------:|:-------------------------:
Point-prompt | ![point-prompt](figs/examples/demo_img_point.png)
Box-prompt |  ![box-prompt](figs/examples/demo_img_box.png)
Segment everything |![segment everything](figs/examples/demo_img_everything.png)

## Model
EfficientTAM checkpoints will be available soon on the [Hugging Face Space](https://huggingface.co/spaces/yunyangx/EfficientTAM/tree/main).

## Acknowledgement

+ [SAM2](https://github.com/facebookresearch/sam2)
+ [SAM2-Video-Predictor](https://huggingface.co/spaces/fffiloni/SAM2-Video-Predictor)
+ [florence-sam](https://huggingface.co/spaces/SkalskiP/florence-sam)
+ [SAM](https://github.com/facebookresearch/segment-anything)
+ [EfficientSAM](https://github.com/yformer/EfficientSAM)

If you're using Efficient Track Anything in your research or applications, please cite using this BibTeX:
```bibtex


@article{xiong2024efficienttam,
  title={Efficient Track Anything},
  author={Yunyang Xiong, Chong Zhou, Xiaoyu Xiang, Lemeng Wu, Chenchen Zhu, Zechun Liu, Saksham Suri, Balakrishnan Varadarajan, Ramya Akula, Forrest Iandola, Raghuraman Krishnamoorthi, Bilge Soran, Vikas Chandra},
  journal={preprint arXiv:2411.18933},
  year={2024}
}
```
