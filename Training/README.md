# Training code

This folder holds the PyTorch code that the ERnet models were trained with. It is research code, kept as it was used, and it has not been run again for this release. Read the list of gaps below before you plan around it.

| Path | Contents |
| --- | --- |
| `archs/swinir_rcab_arch.py` | The single-frame model. The released weights, `20220306_ER_4class_swinir_nch1.pth`, load into `SwinIR_RCAB` with its default arguments, see `load_model` in [`segment.py`](../segment.py) |
| `archs/swin3d_rcab3_arch.py`, `archs/swin3d_rcab.py` | The spatio-temporal model described in the paper, which takes adjacent frames of a time-lapse and uses 3D shifted windows |
| `archs/rcan_arch.py` | The residual channel attention network of the earlier [ERNet](https://github.com/charlesnchr/ERNet) |
| `run.py`, `options.py`, `models.py`, `datahandler.py`, `plotting.py` | Training loop, command line options, model selection, data loading and validation plots |
| `datagen.py` | Cuts pairs of image stacks and label stacks into training crops |
| `Simulate_ER/` | Generator for synthetic ER networks with a known graph, see below |

## Data

The images and labels are public on figshare as [ERnet datasets](https://doi.org/10.6084/m9.figshare.21975878) under CC BY 4.0. `training dataset_1.zip` and `training dataset_2.zip` contain SIM time-lapse stacks (`N.tif`) and their label stacks (`segN.tif`). The labels were made with the Trainable Weka plugin of ImageJ, by tuning a random forest model by hand for each image.

The paper gives the training settings: 650 training and 70 validation crops of 128x128 px, cross-entropy loss over the four classes, the Adam optimiser, 100 epochs with a batch size of 10 and a learning rate that is halved about halfway, on one Nvidia A100.

## Gaps

- `run.py` assumes a CUDA GPU. The `--cpu` flag only covers inference.
- `run.py` imports a module called `testfunctions` that is not in the repository, so it stops at the import. The module is only called when `--testFunction` is set.
- Metrics are logged to Weights & Biases, to a project called `phd`.
- `models.py` builds the spatio-temporal model for `--model swin3d`. There is no option that builds the single-frame model of the released weights.
- `datagen.py` has the paths of the source stacks written into the file.

## Synthetic ER networks

`Simulate_ER/Simulate_ER_images_script.py` draws random networks with a known ground truth. It scatters points, triangulates them, takes the Voronoi tessellation, bends every edge with a cubic spline and renders the result with blur and noise. The ten networks of the paper's ground-truth test were made this way.

```bash
cd Training/Simulate_ER
python Simulate_ER_images_script.py
```

This writes ten pairs of ground truth and noisy image to a dated folder, with plots of the intermediate steps. It needs numpy, scipy, matplotlib, scikit-image and pillow. The parameters, such as image size, number of seed points and noise level, are arguments of `generate_ER_images` at the bottom of the file.
