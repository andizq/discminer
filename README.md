<p align="center">
<img src="https://raw.githubusercontent.com/andizq/andizq.github.io/master/discminer/discminer_logo.jpeg" width="500" height="" ></p>

<h2 align="center">The Channel Map Modelling Code</h2>

<p align="center">

<a href="https://github.com/andizq/discminer/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-FEE440.svg?style=for-the-badge"></a>
<a href="https://github.com/andizq/discminer/pulls"><img alt="Pull request?" src="https://img.shields.io/badge/Become%20a-miner%20%e2%9a%92-00BBF9.svg?style=for-the-badge"></a>
<a href="https://github.com/andizq"><img alt="andizq" src="https://img.shields.io/badge/with%20%e2%99%a1%20by-andizq-ff1414.svg?style=for-the-badge"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge"></a>
</p>


<div align="center">
  Welcome to the discminer repository! Looking for quick examples and tutorials? Check out the docs.
  <br />
  <a href="https://github.com/andizq/discminer/issues/new?assignees=&labels=bug&title=bug%3A+">Report a Bug</a>
  ·
  <a href="https://github.com/andizq/discminer/issues/new?assignees=&labels=enhancement&title=feature%3A+">Request a Feature</a>
  .
  <a href="https://github.com/andizq/discminer/issues/new?assignees=&labels=question&title=question%3A+">Ask a Question</a>
</div>


### Features

- Model intensity channel maps from gas discs using MCMC parameter samplers. 
- Fit intensity and rotation velocity simultaneously. Stellar mass and inclination are no longer degenerate parameters!
- Supports different prescriptions for the disc rotation velocity: Keplerian (2D or 3D) + pressure support + self-gravity.   
- Upper and lower emitting surfaces of your disc can be modelled independently.
- Easy customisation of model parameterisations as a function of the 3D disc coordinates (azimuthal asymmetries are possible!).
- Compatible with input of (irregular) emitting surfaces obtained with the geometrical method of Pinte et al. 2018b.
- Provides tools for the analysis of gas structure and kinematics.
- It has been tested for circumstellar discs but it may be well suited for galactic discs too.


### Mining tools

#### pick

Mine small scale fluctuations in the disc, possibly related to the presence of planets.

#### hammer

Mine larger scale substructure: produce (de-)projected residual maps and illustrations of the gas structure and attributes.

#### rail

Compute azimuthal or radial averages from intensity or kinematical observables.

#### cart

Store model attributes and their default prescriptions.


### Discminer history

Discminer began life as the model.disc2d.py library of [sf3dmodels](https://github.com/andizq/sf3dmodels). This repository is dedicated for versions >1.0 of the code. 

#### v1.0

- Use astropy units wherever possible. All physical parameters provided by the user must be astropy.units instances.
- Creation of several mining tools included in the (1) `pick`, (2) `hammer`, (3) `rail` and (4) `cart` modules to (1) mine small scale fluctuations in the disc, possibly related to the presence of planets; (2) mine larger scale substructure: produce residual maps and illustrations of the gas structure and attributes; (3) compute azimuthal averages from observables; (4) store model attributes and their default prescriptions.  

#### Collaborators

#### License

discminer is published under the [MIT license](https://github.com/andizq/discminer/blob/main/LICENSE).