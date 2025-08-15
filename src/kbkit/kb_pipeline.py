import pandas as pd
from collections import defaultdict

from .kb import KBThermo

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

    def run(self, energy_units="kJ/mol"):
        r"""
        Run Kirkwood-Buff analysis via :class:`kbkit.kb.kb_thermo.KBThermo`.

        Parameters
        ----------
        energy_units: str
            Units of energy for analysis. Defaults to 'kJ/mol'
        """
        self.to_dict(energy_units)

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
    


    