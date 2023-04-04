<p align="center">
<img src="https://raw.githubusercontent.com/andizq/andizq.github.io/master/discminer/discminer_logo.jpeg" width="500" height="" ></p>

<h2 align="center">The Channel Map Modelling Code</h2>

<div align="center">
<a href="https://github.com/andizq/discminer/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-FEE440.svg?style=for-the-badge"></a>
<a href="https://github.com/andizq/discminer/pulls"><img alt="Pull request?" src="https://img.shields.io/badge/Become%20a-miner%20%e2%9a%92-00BBF9.svg?style=for-the-badge"></a>
<a href="https://github.com/andizq"><img alt="andizq" src="https://img.shields.io/badge/with%20%e2%99%a1%20by-andizq-ff1414.svg?style=for-the-badge"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge"></a>
</div>


<div align="center">
  Welcome to the discminer repository! Looking for quick examples and tutorials? Check out the docs.
  <br />
  <a href="https://github.com/andizq/discminer/issues/new?assignees=&labels=bug&title=bug%3A+">Report a Bug</a>
  ·
  <a href="https://github.com/andizq/discminer/issues/new?assignees=&labels=enhancement&title=feature%3A+">Request a Feature</a>
  ·
  <a href="https://github.com/andizq/discminer/issues/new?assignees=&labels=question&title=question%3A+">Ask a Question</a>
</div>


- Model channel maps from molecular line observations of discs by fitting intensity and rotation velocity.
- Compute moment maps, rotation curves and meridional velocities, intensity and line width profiles.
- Quantify deviations from Keplerian motion and investigate their relationship to intensity and line width fluctuations.
- Customise model parameterisations as function of the 3D disc coordinates easily (azimuthal asymmetries are possible!).
- Employ different prescriptions for the disc rotation velocity if needed: Keplerian + pressure support + self-gravity.
- Model upper and lower emitting surfaces of the disc simultaneously.
- Use irregular emitting surfaces obtained with external non-parametric methods if required.
- Analyse the disc physical structure and dynamics using built-in tools.

<img
  src="images/discminer_outline.png"
  alt="Discminer workflow and capabilities"
  style="display: inline-block; margin: 0 auto; max-width: 500px">

### Mining, analysis and visualisation tools

#### rail

- Extract azimuthal and radial profiles of intensity, line width and velocity from moment maps.
- Compute rotation curves and decompose the three-dimensional velocity field of the disc.
- Reveal large-scale substructures and investigate coherence of observable signatures.

#### pick

- Quantify small-scale fluctuations in the disc.
- Reveal localised velocity perturbations and/or sites of enhanced velocity dispersion.

#### moment maps

- Compute moment maps. Available kernels: Gaussian, bell, doubleGaussian, doubleBell.
- Output moments include **peak intensity**, **line width** and **centroid velocity**.

#### channel maps

- Visualise model/data channels, and extract spectra interactively.

#### disc geometry

- Use sky or disc [Cartesian or polar] projections interchangeably.
- Overlay the geometric structure of the disc on moment and channel maps easily. 


## Installation

```bash
pip install discminer
```

### Optional Dependencies

- [termtables](https://pypi.org/project/termtables)
- termplotlib
- FilFinder 
- ipython
- schwimmbad

## How to use

The package documentation is under construction but you can find practical examples showing the main
functionality of `discminer` in the `./template` folder of this repository. To run the examples
on your local machine you can clone this repository and follow the instructions in the readme file,

```bash
git clone https://github.com/andizq/discminer.git
cd discminer/template
less README.rst
```

### Discminer history

`discminer` began life as the model.disc2d.py library of [sf3dmodels](https://github.com/andizq/sf3dmodels).

#### v0.0.x

- Transiting to astropy units.
- Addition of mining tools.

### License

`discminer` is published under the [MIT license](https://github.com/andizq/discminer/blob/main/LICENSE).

### Citation

If you find `discminer` useful for your research please cite the work of [Izquierdo et al. 2021](https://ui.adsabs.harvard.edu/abs/2021A%26A...650A.179I/abstract),

```latex
@ARTICLE{2021A&A...650A.179I,
       author = {{Izquierdo}, A.~F. and {Testi}, L. and {Facchini}, S. and {Rosotti}, G.~P. and {van Dishoeck}, E.~F.},
        title = "{The Disc Miner. I. A statistical framework to detect and quantify kinematical perturbations driven by young planets in discs}",
      journal = {\aap},
     keywords = {planet-disk interactions, planets and satellites: detection, protoplanetary disks, radiative transfer, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2021,
        month = jun,
       volume = {650},
          eid = {A179},
        pages = {A179},
          doi = {10.1051/0004-6361/202140779},
archivePrefix = {arXiv},
       eprint = {2104.09596},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021A&A...650A.179I},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
