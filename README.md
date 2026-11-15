<h1 align="center">ERnet</h1>

<p align="center">
  <b>Turn images of the endoplasmic reticulum into a network you can measure.</b><br>
  A vision transformer labels every pixel as tubule, sheet or sheet-based tubule.<br>
  The tubules become a graph, and the graph puts numbers on the shape of the ER.
</p>

<p align="center">
  <a href="https://doi.org/10.1038/s41592-023-01815-0"><img src="https://img.shields.io/badge/Nature%20Methods-2023-b31b1b" alt="Paper in Nature Methods"></a>
  <a href="https://doi.org/10.1101/2022.05.17.492189"><img src="https://img.shields.io/badge/bioRxiv-2022.05.17.492189-b31b1b" alt="bioRxiv preprint"></a>
  <a href="https://colab.research.google.com/github/charlesnchr/ERnet-v2/blob/main/ERnet.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
  <a href="https://doi.org/10.6084/m9.figshare.21975878"><img src="https://img.shields.io/badge/data-figshare-blue" alt="Datasets on figshare"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="GPL-3.0 license"></a>
</p>

<p align="center">
  <img src="fig/hero.gif" alt="A time-lapse of the endoplasmic reticulum wiping to its segmentation into tubules, sheets and sheet-based tubules, and then to the skeleton graph of the tubules" width="900">
</p>
<p align="center">
  <sub>Three crops of a 30-frame SIM time-lapse of the ER, segmented and converted to a graph by the model in this repository. Run <code>python fig/make_figures.py</code> to reproduce it.</sub>
</p>

## Try it

Get started right away with our Colab notebooks. If you prefer running the code locally, we provide Jupyter notebooks and Python scripts in this repository. Another option for running ERnet is to use it as a plugin in the graphical front-end, Mambio, based on an integration of the Electron UI framework and a Python environment. Mambio is also included in this repository.

<div align="center">
    <a href="https://colab.research.google.com/github/charlesnchr/ERnet-v2/blob/main/ERnet.ipynb">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-colab-small.png" width="15%"/>
    </a>
</div>

### Repository structure
The graphical user interface, Mambio, is provided in the folder `Graphical-App`. Instructions and further information can be found in the corresponding README.

Code for training models based on the ERnet architecture can be found in the folder `Training`, and further information is also in the README file of that folder. A script that generates synthetic ER images can be found in `Training/Simulate_ER`, see the [Synthetic data generation pipeline](#synthetic-data-generation-pipeline-for-testing) section.

### Mambio

To run inference with ERnet locally, we provide the graphical user interface Mambio: Multi-purpose Advanced ML-based Batch Image Operations. Mambio is a more general software that extends beyond image segmentation — there is also a plugin for [ML-SIM](https://github.com/charlesnchr/ML-SIM).

<p>
   <img width="850" src="fig/mambio.png"></a>
</p>
<br>


### Synthetic data generation pipeline for testing

We propose a model for synthesising ground truth ER images with a simple algorithm. First a set of uniformly distributed coordinates are generated. Delaunay triangulation is performed on the coordinates providing a set of triangles. The circumcentre of each triangle becomes a vertex in a Voronoi diagram. The diagram is formed by connecting vertices between adjacent triangles. This procedure is called Voronoi tessellation. Each edge in the undirected graph corresponding to the Voronoi diagram is the perturbed in the following way: the midpoint of an edge is moved in a random direction by a random distance that is smaller than the length of the edge. The displaced midpoint and the two ends of the edge are then used the produce a smooth curve using cubic spline interpolation of the three points. Finally, the network of vertices connected with interpolated curves are processed with a image formation model that applies blur (diffraction) and noise (Poisson noise and Gaussian read-out noise).

<p>
<img width="850" src="fig/ER-synthetic-pipeline.png"></a>
</p>

The steps of this algorithm are illustrated on the figure above. The implementation can be found in `Training/Simulate_ER`, and the script can be used to generate a collection of test images with
```
python Simulate_ER_images_script.py
```
Parameters are given in the file.

### Preceding work

#### Github repository
[ERNet — Residual network](https://github.com/charlesnchr/ERNet)
<br>
Segmentation of Endoplasmic Reticulum microscopy images using modified CNN-based image restoration models.

#### Science Advances publication
_Meng Lu<sup>1</sup>, Francesca W. van Tartwijk<sup>1</sup>, Julie Qiaojin Lin<sup>1</sup>, Wilco Nijenhuis, Pierre Parutto, Marcus Fantham<sup>1</sup>,  __Charles N. Christensen<sup>1,*</sup>__, Edward  Avezov, Christine E. Holt, Alan Tunnacliffe, David Holcman, Lukas C. Kapitein, Gabriele Kaminski Schierle<sup>1</sup>, Clemens F. Kaminski<sup>1</sup>_</br></br>
<sup>1</sup>University of Cambridge, Department of Chemical Engineering and Biotechnology</br>
<sup> *</sup>Author of this repository - GitHub username: [charlesnchr](http://github.com/charlesnchr)

Pre-print: [https://www.biorxiv.org/content/10.1101/2020.01.15.907444v2](https://www.biorxiv.org/content/10.1101/2020.01.15.907444v2)
<br>
Paper: [https://www.science.org/doi/10.1126/sciadv.abc7209](https://www.science.org/doi/10.1126/sciadv.abc7209)
