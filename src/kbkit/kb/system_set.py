import os
import numpy as np
from functools import cached_property
from collections import defaultdict
import types
import itertools
from pathlib import Path

from sympy import comp

from ..properties.system_properties import SystemProperties
from ..unit_registry import load_unit_registry

class SystemSet: 
    """
    A class to manage a set of systems for Kirkwood-Buff analysis.
    This class provides methods to handle system properties, molecule counts, 
    and other thermodynamic properties for a set of systems, including pure components.

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
        anion_list: list = []
    ):

        self.rdf_dir = rdf_dir 
        self.start_time = start_time
        self.ensemble = ensemble.lower()
        self.gamma_integration_type = gamma_integration_type
        self.gamma_polynomial_degree = gamma_polynomial_degree

        self.base_path = base_path
        self.pure_component_path = pure_component_path 
        self.base_systems = base_systems
        self.pure_component_systems = pure_component_systems

        self.ureg = load_unit_registry()
        self.Q_ = self.ureg.Quantity

        self.salt_pairs = [(x, y) for x, y in itertools.product(cation_list, anion_list)]
       

    def _check_valid_path(self, path):
        """Checks if the given path is valid and accessible.
        
        Parameters
        ----------
        path : str
            The path to check.
        
        Returns
        -------
        str
            The absolute path if valid.
        """
        # checks type of path
        if not isinstance(path, str):
            raise TypeError(f"Expected a string path, got {type(path).__name__}: {path}")
        
        # get absolute path
        abs_path = os.path.abspath(path)

        # check if path exists
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Path does not exist: {abs_path}")
        
        # check if path can be accessed
        if not os.access(abs_path, os.R_OK): # R_OK checks for read permissions
            raise PermissionError(f'Cannot access path: {abs_path}')
        
        return abs_path

    @property
    def base_path(self):
        """str: The base path where the systems are located."""
        return self._base_path
    
    @base_path.setter
    def base_path(self, path):
        if not path:
            path = os.getcwd()
        self._base_path = self._check_valid_path(path)

    @property
    def pure_component_path(self):
        """str: The path where pure component systems are located."""
        return self._pure_component_path 
    
    @pure_component_path.setter
    def pure_component_path(self, path):
        if not path:
            # search for pure component directory in parent path
            parent_path = Path(self.base_path).parent 
            matches = [str(p) for p in parent_path.glob("*pure*comp*") if p.is_dir()]
            if matches:
                if len(matches) > 1:
                    print(f"Multiple pure component paths found. Using {matches[0]}.")
                path = matches[0]
            else:
                print(f"WARNING: pure component directory not found. Check that parent directory exists with `pure` and `comp` in name.")
        self._pure_component_path = self._check_valid_path(path)
    
    @property 
    def base_systems(self):
        """list: A list of base systems detected in the base path."""
        return self._base_systems
    
    @base_systems.setter
    def base_systems(self, systems):
        # if systems are provided assign to variable and return
        if systems:
            self._base_systems = systems
            return 
        
        system_dirs = []
        try:
            base_path = Path(self.base_path) # create Path object
            
            # iterate through directories in base path to find directories containing .top file and rdf_dir names
            for path in base_path.iterdir():
                # if not directory move on
                if not path.is_dir():
                    continue
                
                # search for .top
                try:
                    has_top = any(p.suffix == ".top" for p in path.iterdir())
                except PermissionError as e:
                    raise PermissionError(f"Permission denied when listing: {path}") from e 

                # add to directory list if .top file is found
                if has_top:
                    system_dirs.append(path.name)

        except OSError as e:
            raise RuntimeError(f"Error scanning base path '{base_path}': {e}") from e
        
        self._base_systems = system_dirs

    @property 
    def pure_component_systems(self):
        """list: A list of pure component systems detected in the pure component path."""
        return self._pure_component_systems
    
    @pure_component_systems.setter
    def pure_component_systems(self, systems):
        # check if systems were assigned
        if systems:
            self._pure_component_systems = systems
            return
        
        # otherwise parse pure component path for valid systems
        else:
            pcs = []
            # return empty if not an existing path
            if not os.path.isdir(self.pure_component_path):
                return pcs
            
            # create system properties object for each system
            all_systems = {system: SystemProperties(os.path.join(self.base_path, system)) for system in self.base_systems}

            # iterate through each system and append molecule with its _system_temperatures
            temps_by_mol = {}
            for system, props in all_systems.items():
                for mol in props.topology.molecules:
                    temps_by_mol.setdefault(mol, set()).add(props.temperature(units="K"))

            # iterate through systems in pure component path to find systems with one molecule type at temperature
            for d in os.listdir(self.pure_component_path):
                full_path = os.path.join(self.pure_component_path, d)
                if os.path.isdir(full_path):
                    # create object for calculating properties
                    try:
                        props = SystemProperties(full_path)
                    except Exception:
                        continue

                    # skip systems with more than one molecule type present
                    if len(props.topology.molecules) != 1:
                        continue

                    # get molecule and simulation temperature
                    mol = props.topology.molecules[0]
                    temp = props.temperature(units="K")

                    # check that simulation temperature of pure components is close to system temperatures
                    known_temps = temps_by_mol.get(mol, set())
                    if any(np.isclose(temp, t, atol=0.5) for t in known_temps):
                        pcs.append(d)
            
            self._pure_component_systems = pcs


    def sort_systems(self, systems):
        """Sort systems by their mol fraction vectors in ascending order.

        Parameters
        ----------
        systems : list
            List of system names to sort.

        Returns
        -------
        list
            Sorted list of system names based on mol fractions.
        """
        def mol_fr_vector(system):
            """return mol fraction vector for molecules in system"""
            counts = self.system_properties[system].topology.molecule_counts
            total = self.system_properties[system].topology.total_molecules
            return tuple(counts.get(mol, 0) / total for mol in self.top_molecules)
        # apply mol_fr_vector to all systems
        return sorted(systems, key=mol_fr_vector, reverse=False)
    
    @property
    def _systems_set(self):
        """set: Union of base and pure component system names."""
        return set(self.base_systems) | set(self.pure_component_systems or [])
    
    def _filter_systems(self):
        """Filters the systems accordingly into base and pure components"""
        # also filter base_systems if contains pure components
        if '_systems_filtered' not in self.__dict__:
            base_systems = []
            pure_component_systems = []
            multiple_found = {mol: [] for mol in self.top_molecules}
           
            # sort systems accordingly
            for system in self._systems_set:
                top = self.system_properties[system].topology
                n = len(top.molecules)
                # if pure component found, add to pure_component systems
                if n == 1:
                    pure_component_systems.append(system)
                    # tracks number of pure component systems found for each molecule
                    multiple_found[top.molecules[0]].append(system)
                # if more than one component found, check for rdf then add to base systems
                elif n > 1:
                    has_rdf = Path(os.path.join(top.syspath, self.rdf_dir)).is_dir()
                    if has_rdf:
                        base_systems.append(system)
            
            # now filter pure components in case there is a system found in pure component path and base system
            for mol, systems in multiple_found.items():
                if len(systems) > 1:
                    # now filter---assuming the correct system is in base path since that contains other systems of interest
                    for system in systems:
                        if not (Path(self.base_path) / system).exists():
                            pure_component_systems.remove(system)
            
            # check that length of pure components equals length of molecules 
            if len(pure_component_systems) < len(self.top_molecules):
                missing_mols = [mol for mol, systems in multiple_found.items() if len(systems) < 1]
                print(f"WARNING: missing pure component systems for molecules: {missing_mols}")

            # now reassign base and pure component system lists
            self.base_systems = base_systems
            self.pure_component_systems = pure_component_systems
            
            # mark the the systems have been filtered
            self._systems_filtered = True   

    @cached_property
    def systems(self):
        """list: Sorted list of all systems (base + pure components)."""
        # filter systems into base/pure components accordingly
        self._filter_systems()
        # sort systems by compositions and name
        return self.sort_systems(self._systems_set)

    @property
    def n_sys(self):
        """int: The number of systems in the set."""
        count = len(self.systems)
        if count == 0:
            raise ValueError('Number of systems cannot be zero.')
        return count

    @cached_property
    def system_properties(self):
        """dict[str, :class:`kbkit.properties.system_properties.SystemProperties`]: Mapping of system names to SystemProperties objects."""
        props = {}
        # for each system, find its parent dir and create system properties object from path to system dir
        for system in self._systems_set:
            base = self.pure_component_path if system in self.pure_component_systems else self.base_path
            path = os.path.join(base, system)
            props[system] = SystemProperties(path, self.ensemble)
        return props

    @cached_property
    def top_molecules(self):
        """list: A list of unique molecules present in all systems. This encompasses all anions and cations individually, before being combined into salt pairs."""
        # unique molecule names present in systems
        mols_present = set()
        for system in self._systems_set:
            mols_present.update(self.system_properties[system].topology.molecules)
        return list(mols_present)
    
    @property
    def salt_pairs(self):
        """list: List of salt pairs as (cation, anion) tuples."""
        return self._salt_pairs
    
    @salt_pairs.setter
    def salt_pairs(self, pairs):
        # validates the salt_pairs list
        if not isinstance(pairs, list):
            raise TypeError(f"Expected a list of salt pairs, got {type(pairs).__name__}: {pairs}")
        if not all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs):
            raise ValueError("Each salt pair must be a tuple of two elements (cation, anion).")
        # checks molecules in pairs are in top_molecules
        for pair in pairs:
            if not all(mol in self.top_molecules for mol in pair):
                raise ValueError(f"Salt pair {pair} contains molecules not present in top_molecules: {self.top_molecules}")
        self._salt_pairs = pairs
    
    @cached_property
    def nosalt_molecules(self):
        """list: Molecules not part of any salt pair."""
        # filter out molecules that are part of salt pairs
        return [mol for mol in self.top_molecules if mol not in [x for pair in self.salt_pairs for x in pair]]
    
    @cached_property
    def unique_molecules(self):
        """list: Molecules after combining salt pairs as single entries."""
        _gm_molecules = self.nosalt_molecules.copy()
        _gm_molecules.extend(['-'.join(pair) for pair in self.salt_pairs])
        return _gm_molecules

    @property
    def n_comp(self):
        """int: Number of unique components (molecules) in the system set."""
        return len(self.unique_molecules)

    def _system_total_molecules(self):
        """dict: Total molecules count for each system."""
        return {system: self.system_properties[system].topology.total_molecules for system in self.systems}

    def _system_molecule_counts(self):
        """dict: Molecule counts per system as dicts."""
        return {system: self.system_properties[system].topology.molecule_counts for system in self.systems}
    
    def _cached_lookup(self, key, compute_fn):
        """Utility to cache and lookup computed properties.

        Parameters
        ----------
        key : hashable
            Cache key.
        compute_fn : callable or any
            Function to compute the value if not cached, or a precomputed value.

        Returns
        -------
        any
            Cached or computed value.
        """
        if not hasattr(self, '_cache'):
            self._cache = {}
        # check the type of value to store in cache
        if key not in self._cache:
            if isinstance(compute_fn, types.FunctionType):
                self._cache[key] = compute_fn()
            else:
                self._cache[key] = compute_fn
        return self._cache[key]
        
    def _pure_n_elec(self):
        """dict: Number of electrons per molecule in pure component systems."""
        key = ('_pure_n_elec')

        def compute_electron_dict():
            # calculate master electron count dictionary for each unique molecule present
            electron_dict = defaultdict(int)
            # iterate through each system and its topology to find molecules
            for system in self.systems:
                top = self.system_properties[system].topology
                for mol in top.molecules:
                    # Only set if not already set (defaultdict ensures this is safe)
                    if electron_dict[mol] == 0:
                        electron_dict[mol] = top.electron_counts[mol]
            return electron_dict
                        
        return self._cached_lookup(key, compute_electron_dict)

    def _system_mol_fr(self):
        """dict of dict: Mole fraction of each molecule in each system."""
        return {
            system: {mol: count / self._system_total_molecules()[system] for mol, count in mol_counts.items()}
            for system, mol_counts in self._system_molecule_counts().items()
        }

    def _system_temperatures(self, units="K"):
        """dict: Temperatures of systems in specified units.

        Parameters
        ----------
        units : str, optional
            Units for temperature. Default is 'K'.

        Returns
        -------
        dict
            Mapping system name to temperature in given units.
        """        
        key = ("_system_temperature", units)
        return self._cached_lookup(
            key, lambda: {
                system: self.system_properties[system].temperature(start_time=self.start_time, units=units)
                for system in self.systems
            }
        )

    def _system_volumes(self, units="nm^3"):
        """dict: Volumes of systems in specified units.

        Parameters
        ----------
        units : str, optional
            Units for volume. Default is 'nm^3'.

        Returns
        -------
        dict
            Mapping system name to volume in given units.
        """
        key = ("_system_volumes", units)
        return self._cached_lookup(
            key, lambda: {
                system: self.system_properties[system].volume(start_time=self.start_time, units=units)
                for system in self.systems
            }
        )

    def _pure_molar_volumes(self, units="nm^3/molecule"):
        """dict: Molar volumes of pure component molecules in specified units.

        Parameters
        ----------
        units : str, optional
            Units for molar volume (e.g., 'nm^3/molecule'). Default is 'nm^3/molecule'.

        Returns
        -------
        dict
            Mapping molecule to molar volume in given units.
        """
        key = ("_pure_molar_volumes", units)
        # get individual units for molar volume calculation
        V_unit, N_unit = units.split('/')
        return self._cached_lookup(      
            key, lambda: {
                self.system_properties[system].topology.molecules[0]: self._system_volumes(units=V_unit)[system] / self.Q_(self._system_total_molecules()[system], "molecule").to(N_unit).magnitude
                for system in self.pure_component_systems
            }
        )

    def _system_enthalpies(self, units="kJ/mol"):
        """dict: Total system enthalpies in specified units.

        Parameters
        ----------
        units : str, optional
            Units for enthalpy. Default is 'kJ/mol'.

        Returns
        -------
        dict
            Mapping system to enthalpy value.
        """
        key = ("_system_enthalpies", units)
        return self._cached_lookup( 
            key, lambda: {
                system: self.system_properties[system].enthalpy(start_time=self.start_time, units=units)
                for system in self.systems
            }
        )

    def _pure_enthalpies(self, units="kJ/mol"):
        """dict: Pure component enthalpies in specified units.

        Parameters
        ----------
        units : str, optional
            Units for enthalpy. Default is 'kJ/mol'.

        Returns
        -------
        dict
            Mapping molecule to pure component enthalpy.
        """
        key = ("_pure_enthalpies", units)
        return self._cached_lookup( 
            key, lambda: {
                self.system_properties[system].topology.molecules[0]: self._system_enthalpies(units=units)[system]
                for system in self.pure_component_systems
            }
        )
    
    def _system_ideal_mixing_enthalpy(self, units="kJ/mol"):
        """dict: Ideal mixing enthalpy for each system in specified units.

        Parameters
        ----------
        units : str, optional
            Units for enthalpy. Default is 'kJ/mol'.

        Returns
        -------
        dict
            Mapping system to ideal mixing enthalpy.
        """
        key = ("_system_ideal_mixing_enthalpy", units)
        return self._cached_lookup(
            key, lambda: {
                system: sum(self._system_mol_fr().get(system, {})[mol] * self._pure_enthalpies(units=units).get(mol, 0) for mol in self._system_mol_fr().get(system, {}))
                for system in self.systems
            }
        )
    
    def _system_mixing_enthalpy(self, units="kJ/mol"):
        """dict: Mixing enthalpy (excess enthalpy) for each system in specified units.

        Parameters
        ----------
        units : str, optional
            Units for enthalpy. Default is 'kJ/mol'.

        Returns
        -------
        dict
            Mapping system to mixing enthalpy.
        """
        key = ("_system_mixing_enthalpy", units)
        return self._cached_lookup(
            key, lambda: {
                system: self._system_enthalpies(units=units).get(system, 0) - self._system_ideal_mixing_enthalpy(units=units).get(system, 0)
                for system in self.systems
            }
        )

    @cached_property
    def top_mol_fr(self):
        """np.ndarray: Mol fraction array for each molecule in each system."""
        return np.array([
            [mfr.get(mol, 0) for mol in self.top_molecules]
            for system, mfr in self._system_mol_fr().items()
        ])
    
    @cached_property
    def mol_fr(self):
        """np.ndarray: Mol fraction array including salt pairs."""
        mfr = np.zeros((self.n_sys, self.n_comp))
        # iterate through systems an molecules in topology to find and adjust for salt pairs
        for i, system in enumerate(self.systems):
            for j, mol in enumerate(self.top_molecules):
                # check if molecule is a salt pair
                mol_split = mol.split('-')
                # Handle salt pairs
                if len(mol_split) > 0 and mol in self.salt_pairs:
                    for salt in mol_split:
                        k = list(self.unique_molecules).index(mol_split)
                        mfr[i, k] += self._system_mol_fr().get(system, {}).get(salt, 0)
                else:
                    # Handle single molecules
                    mfr[i, j] += self._system_mol_fr().get(system, {}).get(mol, 0)
        return mfr
    
    @cached_property
    def total_molecules(self):
        """np.ndarray: Total number of molecules in each system."""
        return np.fromiter(self._system_total_molecules().values(), dtype=int)
    
    @cached_property
    def molecule_counts(self):
        """np.ndarray: Molecule counts for each molecule in each system."""
        return np.array([
            [counts.get(mol, 0) for mol in self.top_molecules]
            for system, counts in self._system_molecule_counts().items()
        ])
    
    def n_elec(self):
        """
        Returns
        -------
        np.ndarray
            Number of electrons in each unique molecule.
        """
        return np.array([self._pure_n_elec()[mol] for mol in self.top_molecules])
    
    def n_elec_bar(self):
        r"""
        Linear combination of number of electrons for each system.

        Returns
        -------
        np.ndarray
            A linear combination of number of electrons in each system.
            
        Notes
        -----
        This is calculated as the dot product of the mol fraction vector and the number of electrons vector.

        .. math::
           \overline{Z} = \sum_{i=1}^n x_i Z_i
        
        where:
            - :math:`x_i` is the mol fraction of molecule :math:`i`
            - :math:`Z_i` is the number of electrons in molecule :math:`i`
        """
        return self.top_mol_fr @ self.n_elec()
    
    def delta_n_elec(self):
        r"""
        Difference in number of electrons between each molecule and the last molecule.

        Returns
        -------
        np.ndarray
            A 1D array of the difference in number of electrons between each molecule and the last molecule.
            
        Notes
        -----
        This is calculated as the number of electrons in each molecule minus the number of electrons in the last molecule.
        
        .. math::
            \Delta Z_i = Z_i - Z_n
            
        where:
            - :math:`N_i^e` is the number of electrons in molecule :math:`i` 
            - :math:`N_n^e` is the number of electrons in the last molecule
        """
        return self.n_elec()[:-1] - self.n_elec()[-1]

    def T(self, units="K"):
        r"""
        Temperatures of systems in specified units.

        Parameters
        ----------
        units : str, optional
            Units for temperature. Default is 'K'.
        
        Returns
        -------
        np.ndarray
            A 1D array of temperatures for each system in specified units.
        """
        return np.fromiter(self._system_temperatures(units=units).values(), dtype=float)
    
    def volume(self, units="nm^3"):
        r"""
        Volume of simulation boxes for each system in specified units.

        Parameters
        ----------
        units : str, optional
            Units for volume. Default is 'nm^3'.
        
        Returns
        -------
        np.ndarray
            A 1D array of volumes for each system in specified units.
        """
        return np.fromiter(self._system_volumes(units=units).values(), dtype=float)

    def rho(self, units="molecule/nm^3"):
        r"""
        Number density of molecules in each system in specified units.

        Parameters
        ----------
        units : str, optional
            Units for number density. Default is 'molecule/nm^3'.

        Returns
        -------
        np.ndarray
            A 2D array of number densities for each molecule in each system in specified units.
        """
        n_units, v_units = units.split('/') # get the target units
        N = self.mol_fr * self.total_molecules[:, np.newaxis] #  calculate number of molecules
        N = self.Q_(N, "molecule").to(n_units).magnitude # convert to desired units
        V = self.volume(units=v_units)[:,np.newaxis] # get total system volume
        return N / V
    
    def molar_volume(self, units="nm^3/molecule"):
        r"""
        Molar volume of each molecule in specified units.

        Parameters
        ----------
        units : str, optional
            Units for molar volume. Default is 'nm^3/molecule'. 
        
        Returns
        -------
        np.ndarray
            A 1D array of molar volumes for each molecule in specified units.
        """
        return np.array([self._pure_molar_volumes(units)[mol] for mol in self.top_molecules])
    
    def delta_V(self, units="nm^3/molecule"):
        r"""
        Difference in molar volumes between each molecule and the last molecule in specified units.

        Parameters
        ----------
        units : str, optional
            Units for molar volume. Default is 'nm^3/molecule'. 
        
        Returns
        -------
        np.ndarray
            A 1D array of the difference in molar volumes between each molecule and the last molecule.

        Notes
        -----
        This is calculated as the molar volume of each molecule minus the molar volume of the last molecule.
        
        .. math::
            \Delta V_i = V_i - V_n
            
        where:
            - :math:`V_i` is the molar volume of molecule :math:`i` 
            - :math:`V_n` is the molar volume of the last molecule
        """
        return self.molar_volume(units)[:-1] - self.molar_volume(units)[-1]
    
    def V_bar(self, units="nm^3/molecule"):
        """
        Linear combination of molar volumes for each system in specified units.

        Parameters
        ----------
        units : str, optional
            Units for molar volume. Default is 'nm^3/molecule'.
        
        Returns
        -------
        np.ndarray
            A linear combination of molar volumes in specified units.

        Notes
        -----
        This is calculated as the dot product of the mol fraction vector and the molar volume vector.

        .. math::
           \overline{V} = \sum_{i=1}^n x_i V_i
        
        where:
            - :math:`x_i` is the mol fraction of molecule :math:`i`
            - :math:`V_i` is the molar volume of molecule :math:`i`
        """
        return self.top_mol_fr @ self.molar_volume(units=units)

    def rho_ij(self, units="molecule/nm^3"):
        r"""
        Pairwise number density of molecules in each system in specified units.

        Parameters
        ----------
        units : str, optional
            Units for number density. Default is 'molecule/nm^3'.   
        
        Returns
        -------
        np.ndarray
            A 3D array of pairwise number densities for each molecule in each system in specified units.
        
        Notes
        -----
        This is calculated as the outer product of the number density vector with itself, resulting in a 3D array where each slice corresponds to a system.

        .. math::
            \rho_{ij} = \rho_i \cdot \rho_j
        
        where:
            - :math:`\rho_i` is the number density of molecule :math:`i`
            - :math:`\rho_j` is the number density of molecule :math:`j`
        """
        return self.rho(units=units)[:, :, np.newaxis] * self.rho(units=units)[:, np.newaxis, :]
    

