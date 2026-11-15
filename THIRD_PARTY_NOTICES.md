# Third-party code

ERnet is released under the GNU General Public License v3.0, see [LICENSE](LICENSE). Some files are derived from other open-source projects. Their licenses allow redistribution under the GPL, and the notices below are kept as those licenses require.

| File | Derived from | License of the original |
| --- | --- | --- |
| `Training/archs/swinir_rcab_arch.py`, `Graphical-App/pyengine/pysrc/swinir_rcab_arch.py` | [SwinIR](https://github.com/JingyunLiang/SwinIR) by Jingyun Liang et al., which builds on the [Swin Transformer](https://github.com/microsoft/Swin-Transformer) by Ze Liu et al. Weight initialisation and stochastic depth come from [timm](https://github.com/huggingface/pytorch-image-models) by Ross Wightman. | Apache-2.0 (SwinIR, timm), MIT (Swin Transformer) |
| `Training/archs/swin3d_rcab.py`, `Training/archs/swin3d_rcab3_arch.py`, `Graphical-App/pyengine/pysrc/swin3d_rcab3_arch.py` | [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) by Ze Liu et al. | Apache-2.0 |
| `Training/archs/rcan_arch.py` and the RCAN classes in `Training/models.py` | [RCAN](https://github.com/yulunzhang/RCAN) by Yulun Zhang et al., which builds on [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch) | No license file in the RCAN repository, MIT for EDSR-PyTorch |
| `NonLocalBlock2D` in `Training/models.py` | [Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch) by AlexHex7 | Apache-2.0 |
| The U-Net blocks in `Training/models.py` | [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) by milesial | GPL-3.0 |
| `Training/Simulate_ER/delaunay2D.py` | [pyDelaunay2D](https://github.com/jmespadero/pyDelaunay2D) by Jose M. Espadero, based on code by Ayron Catteau | GPL-3.0 |

The files from Apache-licensed projects were changed for this project. The main changes are the segmentation head, the channel attention blocks inside the Swin layers and the single-channel input. The text of the Apache License 2.0 is in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

The graph analysis uses [sknw](https://github.com/Image-Py/sknw) (BSD-3-Clause) and [NetworkX](https://networkx.org/) (BSD-3-Clause) as installed packages. Their code is not copied into this repository.
