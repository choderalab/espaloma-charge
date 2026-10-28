Espaloma Charge
=======

[![CI](https://github.com/choderalab/espaloma_charge/actions/workflows/ci.yml/badge.svg)](https://github.com/choderalab/espaloma_charge/actions/workflows/ci.yml)

Standalone charge assignment from Espaloma framework. https://doi.org/10.1039/D2SC02739A

## Installation

### conda-forge

We recomend installing `espaloma_charge` via conda-forge:

```bash
$ mamba create -n espaloma -c conda-forge espaloma_charge
```

If you plan on using [`openff-toolkit`](https://github.com/openforcefield/openff-toolkit), then install that as well when creating your `espaloma_charge` environment for optimal dependency solving:

```bash
$ mamba create -n espaloma -c conda-forge espaloma_charge openff-toolkit
```

### pypi

We also have `espaloma_charge` on pypi:

```bash
$ pip install torch torch-geometric espaloma_charge
```

## Examples
**Option0: Assign charges to rdkit molecule.**

```python
>>> from rdkit import Chem; from espaloma_charge import charge
>>> molecule = Chem.MolFromSmiles("N#N")
>>> charge(molecule)
array([0., 0.], dtype=float32)

```

Assign charges to your favorite molecule in 
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1e14EkNyidPI0wXBGcewh9m9LC1imSRWZ?usp=sharing)


**Option1: Use with [`openff-toolkit`](https://github.com/openforcefield/openff-toolkit)**(installation required)

```python
>>> from openff.toolkit.topology import Molecule
>>> from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
>>> toolkit_registry = EspalomaChargeToolkitWrapper()
>>> molecule = Molecule.from_smiles("N#N")
>>> molecule.assign_partial_charges('espaloma-am1bcc', toolkit_registry=toolkit_registry)
>>> molecule.partial_charges
<Quantity([0. 0.], 'elementary_charge')>
```

**Option2: Use as Command Line Interface to write [`antechamber`](http://ambermd.org/antechamber/ac.html)-compatible charges.**
```bash
$ espaloma_charge -i in.mol2 -o in.crg
$ antechamber -fi mol2 -fo mol2 -i in.mol2 -o out.mol2 -c rc -cf in.crg 
```

## Development

For a reproducible development environment, we use [`pixi`](https://pixi.sh). It mixes conda-forge packages (`rdkit`, `openff-toolkit`, `pytorch`) with a PyPI `torch-geometric`, and installs `espaloma_charge` itself in editable mode.

```bash
$ pixi install -e dev
$ pixi run -e dev pytest -v espaloma_charge
```

## Training data

A minimal dataset for training the model can be found in https://doi.org/10.5281/zenodo.17308526

## Training instructions

Coming soon.

## Reference
If you are using this little tool in your pipeline, please consider citing:

```
@Article{D2SC02739A,
author ="Wang, Yuanqing and Fass, Josh and Kaminow, Benjamin and Herr, John E. and Rufa, Dominic and Zhang, Ivy and Pulido, Iván and Henry, Mike and Bruce Macdonald, Hannah E. and Takaba, Kenichiro and Chodera, John D.",
title  ="End-to-end differentiable construction of molecular mechanics force fields",
journal  ="Chem. Sci.",
year  ="2022",
volume  ="13",
issue  ="41",
pages  ="12016-12033",
publisher  ="The Royal Society of Chemistry",
doi  ="10.1039/D2SC02739A",
url  ="http://dx.doi.org/10.1039/D2SC02739A"}


@misc{https://doi.org/10.48550/arxiv.2302.06758,
  doi = {10.48550/ARXIV.2302.06758},
  
  url = {https://arxiv.org/abs/2302.06758},
  
  author = {Wang, Yuanqing and Pulido, Iván and Takaba, Kenichiro and Kaminow, Benjamin and Scheen, Jenke and Wang, Lily and Chodera, John D.},
  
  keywords = {Machine Learning (cs.LG), Chemical Physics (physics.chem-ph), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Physical sciences, FOS: Physical sciences},
  
  title = {EspalomaCharge: Machine learning-enabled ultra-fast partial charge assignment},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {Creative Commons Attribution 4.0 International}
}


```
