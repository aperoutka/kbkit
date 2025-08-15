import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

plt.style.use(Path(__file__).parent / 'presentation.mplstyle')


from .kb import KBThermo
from .plotter import Plotter

class KBPipeline:
    """
    A pipeline for performing Kirkwood-Buff analysis of molecular simulations.
    Includes support for running analysis and plotting thermodynamic functions.

    Parameters
    ----------
    base_path : str, optional
        The base path where the systems are located. Defaults to the current working directory. 
    pure_component_path : str, optional
        The path where pure component systems are located. Defaults to a 'pure_components' directory next to the base path.
    base_systems : list, optional
        A list of base systems to include. If not provided, it will automatically detect systems in the base path.
    pure_component_systems : list, optional
        A list of pure component systems to include. If not provided, it will automatically detect systems in the pure component path.
    rdf_dir : str, optional
        The directory name where RDF files are stored. Defaults to 'rdf_files'.
    start_time : int, optional
        The starting time for analysis, used in temperature and enthalpy calculations. Defaults to 0.
    ensemble : str, optional
        The ensemble type for the systems, e.g., 'npt', 'nvt'. Defaults to 'npt'.
    gamma_integration_type : str, optional
        The type of integration to use for gamma calculations. Defaults to 'numerical'.
    gamma_polynomial_degree : int, optional
        The degree of the polynomial for gamma integration. Defaults to 5.
    cation_list : list, optional
        A list of cation names to consider for salt pairs. Defaults to an empty list.
    anion_list : list, optional
        A list of anion names to consider for salt pairs. Defaults to an empty list.
    x_mol: str, optional
        Molecule to use for labeling x-axis in figures for binary systems. Defaults to first element in molecule list.
    molecule_map: dict[str, str], optional.
        Dictionary of molecule ID in topology mapped to molecule names for figure labeling. Defaults to using molecule names in topology.
    """
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
        gamma_polynomial_degree: int = 5,
        cation_list: list = [],
        anion_list: list = [],
        x_mol: str = None,
        molecule_map: dict = None
    ):
        # create KBThermo object
        self.kb = KBThermo(
            base_path=base_path,
            pure_component_path=pure_component_path,
            base_systems=base_systems,
            pure_component_systems=pure_component_systems,
            rdf_dir=rdf_dir,
            start_time=start_time,
            ensemble=ensemble,
            gamma_integration_type=gamma_integration_type,
            gamma_polynomial_degree=gamma_polynomial_degree,
            cation_list=cation_list,
            anion_list=anion_list
        )

        self.x_mol = x_mol 
        self.molecule_map = molecule_map

    def to_dict(self, energy_units="kJ/mol"):
        r"""
        Create a dictionary of properties calculated from :class:`kbkit.kb.kb_thermo.KBThermo`.

        Parameters
        ----------
        energy_units: str
            Units of energy for analysis. Defaults to 'kJ/mol'
        
        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of thermodynamic properties from Kirkwood-Buff analysis
        """
        # returns dictionary of kb properties
        return self.kb._property_map(energy_units)

    def run(self, energy_units="kJ/mol"):
        r"""
        Run Kirkwood-Buff analysis via :class:`kbkit.kb.kb_thermo.KBThermo`.

        Parameters
        ----------
        energy_units: str
            Units of energy for analysis. Defaults to 'kJ/mol'
        """
        self.to_dict(energy_units)

    def to_dataframe(self, name=None, energy_units="kJ/mol"):
        r"""
        Create a pandas dataframe of properties calculated from :class:`kbkit.kb.kb_thermo.KBThermo`.

        Parameters
        ----------
        name: str
            Property to convert into pandas.DataFrame. Includes '`thermo`', which will return results of free energy, enthalpy, and entropy.
        energy_units: str
            Units of energy for analysis. Defaults to 'kJ/mol'
        
        Returns
        -------
        pd.DataFrame
            DataFrame of thermodynamic properties from Kirkwood-Buff analysis
        """
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

        # if property not specified add all properties to dataframe
        if name is None:
            for key in thermo_dict:
                if thermo_dict[key].ndim == 1:
                    _dict[key] = thermo_dict[key]
                elif thermo_dict[key].ndim == 2:
                    _dict.update({f"{key}_{mol}": thermo_dict[key][:,i] for i, mol in enumerate(self.kb.unique_molecules)})
        
        # only add properties with key in thermo_keys to dataframe
        elif any(x in name_lower for x in ['thermo', 'gibbs']):
            _dict.update({key: thermo_dict[key] for key in thermo_keys if key in thermo_dict})
        
        # only add properties with key in saxs_keys to dataframe
        elif any(x in name_lower for x in ['saxs', 'i0', 'io']):
            _dict.update({key: thermo_dict[key] for key in saxs_keys if key in thermo_dict})

        # search for which properties match name and add to dataframe accordingly
        elif (match := next((k for k in (name_lower, name_upper) if k in thermo_dict), None)) is not None:
            if thermo_dict[match].ndim == 1:
                _dict[match] = thermo_dict[match]
            elif thermo_dict[match].ndim == 2:
                _dict.update({f"{match}_{mol}": thermo_dict[match][:,i] for i, mol in enumerate(self.kb.unique_molecules)})
            else:
                print(f"WARNING: number of dimensions {thermo_dict[match].ndim} exceeded 2 for {match}.")

        else:
            raise ValueError(f"Unknown property or name category: '{name}'")
        
        return pd.DataFrame(dict(_dict)) # create pandas.dataframe object  
    

    @property
    def plotter(self):
        """Plotter: Instance of Plotter class (:class:`kbkit.plotter.Plotter`) for creating figures"""
        if not hasattr(self, '_plotter'):
            self._plotter = Plotter(kb_obj=self.kb, x_mol=self.x_mol, molecule_map=self.molecule_map)
        return self._plotter 


    def plot_system(self, name, system=None, units=None, show=True, **kwargs):
        r"""
        Create a plot, either RDF or KBI results, for a specific system.

        Parameters
        ----------
        name: str
            Property to plot. Options: 'rdf', 'kbi'.
        system: str
            Name of system to plot.
        units: str, optional
            Units to plot KBI results in. Default: 'cm^3/mol'.
        show: bool, optional
            Display figure. Default True.
        """
        # returns figure for different kb properties as a function of composition
        if system:
            # plot system rdfs
            if name.lower() in ['rdf', 'gr', 'g(r)']:
                self.plotter.plot_system_rdf(system=system, line=True, show=show, **kwargs)
            
            # plot system kbis
            elif name.lower() in ['kbi', 'kbis', 'kbi_analysis', 'kbianalysis', 'kb_analysis']:
                self.plotter.plot_system_kbi_analysis(system=system, units=units, show=show, **kwargs)
            
            else:
                print('WARNING: Invalid plot option specified! System specific include rdf and kbi analysis.')
        
        else:
            self.plotter.plot_thermo_property(thermo_property=name, units=units, show=show, **kwargs)


    def make_figures(self, energy_units="kJ/mol"):    
        r"""
        Create all figures for Kirkwood-Buff analysis.

        Parameters
        ----------
        energy_units: str
            Energy units for calculations. Default is 'kJ/mol'.
        """
        # create figure for rdf/kbi analysis
        self.plotter.plot_rdf_kbis(show=False)
        # plot KBI as a function of composition
        self.plotter.plot_kbis(units="cm^3/mol", show=False)
        
        # create figures for properties independent of component number
        for thermo_prop in ["lngamma", "dlngamma", "i0", "det_h"]:
            self.plotter.plot_thermo_property(thermo_property=thermo_prop, units=energy_units, show=False)
        
        # plot polynomial fits to activity coefficient derivatives if polynomial integration is performed
        if self.kb.gamma_integration_type == "polynomial":
            for thermo_prop in ["lngamma_fits", "dlngamma_fits"]:
                self.plotter.plot_thermo_property(thermo_property=thermo_prop, units=energy_units, show=False)
        
        # for binary systems plot mixing and excess energy contributions
        if self.kb.n_comp == 2:
            for thermo_prop in ["mixing", "excess"]:
                self.plotter.plot_thermo_property(thermo_property=thermo_prop, units=energy_units, show=False)
        
        # for ternary system plot individual energy contributions on separate figure
        elif self.kb.n_comp == 3:
            for thermo_prop in ["ge", "gm", "hmix", "se"]:
                self.plotter.plot_thermo_property(thermo_property=thermo_prop, units=energy_units, show=False)

        else:
            raise ValueError(f"Additional plotting is not supported for n > 3")