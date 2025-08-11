KBkit: A Python-based toolkit for Kirkwood-Buff Analysis from Molecular Dynamics Simulations
==============================================================================================

.. toctree::
    :maxdepth: 4
    :caption: API Reference:
    :titlesonly:

    kbkit.properties
    kbkit.system_properties
    kbkit.kb 
    kbkit.kb_pipeline
    kbkit.plotter
    examples


Installation
-------------
.. image:: http://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
    :target: https://kbkit.readthedocs.io/
    :alt: docs
.. image:: http://img.shields.io/badge/License-MIT-blue.svg
    :target: https://tldrlegal.com/license/mit-license
    :alt: license
.. image:: https://img.shields.io/badge/Python-3.12%2B-blue

``kbkit`` can be installed from cloning github repository.

.. code-block:: text

    git clone https://github.com/aperoutka/kbkit.git

Creating an anaconda environment with ``kbkit`` dependencies and install ``kbkit``.

.. code-block:: text
    
    cd kbkit
    conda create --name kbkit python=3.12 --file requirements.txt
    conda activate kbkit
    pip install .


File Organization
------------------

.. code-block:: text
    :caption: KB Analysis File Structure

    kbi_dir/
    ├── project/
    │   └── system/
    │       ├── rdf_dir/
    │       │   ├── mol1_mol1.xvg
    │       │   ├── mol1_mol2.xvg
    │       │   └── mol1_mol2.xvg
    │       ├── system_npt.edr
    │       ├── system_npt.gro
    │       └── system.top
    └── pure_components/
        └── molecule1/
            ├── molecule1_npt.edr
            └── molecule1.top

Indices and tables
===================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`