import enum
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
plt.style.use(Path(__file__).parent / "plt_format.mplstyle")

from .kb import KBThermo
from .plotter import Plotter

class KBPipeline:
    # object to access various properties of & run KB analysis
    def __init__(
        self,
        base_path: str = None,
        pure_component_path: str = None,
        base_systems: list = None, 
        pure_component_systems: list = None,
        rdf_dir: str = 'rdf_files',
        start_time: int = 0,
        ensemble: str = 'npt',
        gamma_integration_type: str = 'numerical',
        gamma_polynomial_degree: int = 5
    ):
        self.kb = KBThermo(
            base_path=base_path,
            pure_component_path=pure_component_path,
            base_systems=base_systems,
            pure_component_systems=pure_component_systems,
            rdf_dir=rdf_dir,
            start_time=start_time,
            ensemble=ensemble,
            gamma_integration_type=gamma_integration_type,
            gamma_polynomial_degree=gamma_polynomial_degree
        )

    def to_dict(self, energy_units="kJ/mol"):
        # returns dictionary of kb properties
        return self.kb._property_map(energy_units)

    def run(self, energy_units="kJ/mol"):
        self.to_dict(energy_units)

    def to_dataframe(self, name=None, energy_units="kJ/mol"):
        # returns dataframe of properties from kb analysis
        name_lower = name.lower() if name else None
        name_upper = name.upper() if name else None 

        _dict = defaultdict(dict)

        # add mol frac columns
        _dict.update({f"x_{mol}": self.kb.mol_fr[:,i] for i, mol in enumerate(self.kb.unique_molecules)})

        # Get all thermodynamic data
        thermo_dict = self.to_dict(energy_units)

        # Define keys for groupings
        thermo_keys = {'GE', 'GID', 'GM', 'SE', 'HE'}
        saxs_keys = {'I0'}

        if name is None:
            for key in thermo_dict:
                if thermo_dict[key].ndim == 1:
                    _dict[key] = thermo_dict[key]
                elif thermo_dict[key].ndim == 2:
                    _dict.update({f"{key}_{mol}": thermo_dict[key][:,i] for i, mol in enumerate(self.kb.unique_molecules)})
        
        elif any(x in name_lower for x in ['thermo', 'gibbs']):
            _dict.update({key: thermo_dict[key] for key in thermo_keys if key in thermo_dict})
        
        elif any(x in name_lower for x in ['saxs', 'i0', 'io']):
            _dict.update({key: thermo_dict[key] for key in saxs_keys if key in thermo_dict})

        elif (match := next((k for k in (name_lower, name_upper) if k in thermo_dict), None)) is not None:
            if thermo_dict[match].ndim == 1:
                _dict[match] = thermo_dict[match]
            elif thermo_dict[match].ndim == 2:
                _dict.update({f"{match}_{mol}": thermo_dict[match][:,i] for i, mol in enumerate(self.kb.unique_molecules)})
            else:
                print(f"WARNING: number of dimensions {thermo_dict[match].ndim} exceeded 2 for {match}.")

        else:
            raise ValueError(f"Unknown property or name category: '{name}'")
        
        return pd.DataFrame(dict(_dict))       


    def plot(self, name, x_mol=None, system=None, units=None, show=True, **kwargs):
        # returns figure for different kb properties as a function of composition
        x_mol = x_mol if x_mol is not None else self.kb.unique_molecules[0]
        plotter = Plotter(kb_obj=self.kb, x_mol=x_mol)

        if system:
            if name.lower() in ['rdf', 'gr', 'g(r)']:
                plotter.plot_system_rdf(system=system, line=True, show=show, **kwargs)
            
            elif name.lower() in ['kbi', 'kbis', 'kbi_analysis', 'kbianalysis', 'kb_analysis']:
                plotter.plot_system_kbi_analysis(system=system, units=units, show=show, **kwargs)
            
            else:
                print('WARNING: Invalid plot option specified! System specific include rdf and kbi analysis.')
        
        else:
            plotter.plot_thermo_property(thermo_property=name, units=units, show=show, **kwargs)


    def make_figures(self, x_mol=None, energy_units="kJ/mol"):    
        # make all figures
        x_mol = x_mol if x_mol is not None else self.kb.unique_molecules[0]
        plotter = Plotter(kb_obj=self.kb, x_mol=x_mol)
        plotter.plot_rdf_kbis(show=False)
        
        for thermo_prop in ["lngamma", "dlngamma", "i0", "det_h"]:
            plotter.plot_thermo_property(thermo_property=thermo_prop, units=energy_units, show=False)
        
        if self.kb.gamma_integration_type == "polynomial":
            for thermo_prop in ["lngamma_fits", "dlngamma_fits"]:
                plotter.plot_thermo_property(thermo_property=thermo_prop, units=energy_units, show=False)
        
        if self.kb.n_comp == 2:
            for thermo_prop in ["mixing", "excess"]:
                plotter.plot_thermo_property(thermo_property=thermo_prop, units=energy_units, show=False)
        
        elif self.kb.n_comp == 3:
            for thermo_prop in ["ge", "gm", "hmix", "se"]:
                plotter.plot_thermo_property(thermo_property=thermo_prop, units=energy_units, show=False)

