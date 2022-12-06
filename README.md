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


### Features

- Model channel maps from molecular line observations of discs by fitting intensity and rotation velocity simultaneously.
- Supports different prescriptions for the disc rotation velocity: Keplerian (2D or 3D) + pressure support + self-gravity.   
- Upper and lower emitting surfaces of your disc can be modelled independently.
- Easy customisation of model parameterisations as function of the 3D disc coordinates (azimuthal asymmetries are possible!).
- Can use external emitting surfaces obtained with e.g. the geometrical method of Pinte et al. 2018.
- Provides analysis tools to study the gas disc physical structure and kinematics.


### Mining tools

#### pick

Reveal small scale fluctuations in the disc, possibly related to the presence of planets.

#### hammer (in prep)

Reveal large scale substructures; compute residual maps from line profile properties and investigate coherence of signatures.

#### rail

Compute azimuthal or radial averages from intensity or kinematical observables.

#### cart

Stores model attributes and their default prescriptions.

## Installation

Type in a terminal:

```bash
git clone https://github.com/andizq/discminer.git
cd discminer
python setup.py install
```

### Required dependencies

- [spectral-cube](https://spectral-cube.readthedocs.io/en/latest/installing.html)
- [radio-beam](https://radio-beam.readthedocs.io/en/latest/install.html)
- [emcee](https://emcee.readthedocs.io/en/stable/user/install/)

### Discminer history

Discminer began life as the model.disc2d.py library of [sf3dmodels](https://github.com/andizq/sf3dmodels).

See changes in `CHANGES.md`.

#### v1.0

- Migrating to astropy units.
- Addition of analysis tools for mining.

#### Collaborators

#### License

discminer is published under the [MIT license](https://github.com/andizq/discminer/blob/main/LICENSE).