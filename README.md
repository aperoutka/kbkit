# KBkit: A Python-based toolkit for Kirkwood-Buff Analysis from Molecular Dynamics Simulations

[![docs](http://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://kbkit.readthedocs.io/)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://tldrlegal.com/license/mit-license)
![python 3.12](https://img.shields.io/badge/Python-3.12%2B-blue)

`kbkit` is a `Python` library designed to streamline the analysis of molecular dynamics simulations, focusing on the application of Kirkwood-Buff (KB) theory for the calculation of activity coefficients and excess thermodynamic properties.

## Installation

`kbkit` can be installed from cloning github repository.

```python
git clone https://github.com/aperoutka/kbkit.git
```

Creating an anaconda environment with dependencies and install `kbkit`.

```python
cd kbkit
conda create --name kbkit python=3.12 --file requirements.txt
conda activate kbkit
pip install .
```
