import os
import numpy as np
from functools import cached_property
from collections import defaultdict
import types
import itertools

from ..system_properties import SystemProperties
from ..unit_registry import load_unit_registry

class SystemSet: # add support to catch if file organization is not correct.
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
        """checks if path is a valid path & that reading permisisons exist"""

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
        return self._base_path
    
    @base_path.setter
    def base_path(self, path):
        if not path:
            path = os.getcwd()
        self._base_path = self._check_valid_path(path)

    @property
    def pure_component_path(self):
        return self._pure_component_path 
    
    @pure_component_path.setter
    def pure_component_path(self, path):
        if not path:
            path = os.path.join(os.path.dirname(self.base_path), 'pure_components')
        self._pure_component_path = self._check_valid_path(path)
    
    @property 
    def base_systems(self):
        return self._base_systems
    
    @base_systems.setter
    def base_systems(self, systems):
        if systems:
            self._base_systems = systems
        else:
            system_dirs = []
            for item in os.listdir(self.base_path):
                full_path = os.path.join(self.base_path, item)
                if os.path.isdir(full_path):
                    has_top = any(f.endswith('.top') for f in os.listdir(full_path))
                    has_rdf = os.path.isdir(os.path.join(full_path, self.rdf_dir))
                    if has_top or has_rdf:
                        system_dirs.append(item)
            self._base_systems = system_dirs
    
    @property 
    def pure_component_systems(self):
        return self._pure_component_systems
    
    @pure_component_systems.setter
    def pure_component_systems(self, systems):
        if systems:
            self._pure_component_systems = systems
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
                    try:
                        props = SystemProperties(full_path)
                    except Exception:
                        continue

                    if len(props.topology.molecules) != 1:
                        continue

                    mol = props.topology.molecules[0]
                    temp = props.temperature(units="K")

                    known_temps = temps_by_mol.get(mol, set())
                    if any(np.isclose(temp, t, atol=0.5) for t in known_temps):
                        pcs.append(d)
            
            self._pure_component_systems = pcs


    def sort_systems(self, systems):
        # Sort systems by mol fraction values in order of unique_molecules
        def mol_fr_vector(system):
            counts = self.system_properties[system].topology.molecule_counts
            total = self.system_properties[system].topology.total_molecules
            return tuple(counts.get(mol, 0) / total for mol in self.top_molecules)
        return sorted(systems, key=mol_fr_vector, reverse=False)
    
    @property
    def _systems_set(self):
        return set(self.base_systems) | set(self.pure_component_systems or [])

    @cached_property
    def systems(self):
        return self.sort_systems(self._systems_set)

    @property
    def n_sys(self):
        count = len(self.systems)
        if count == 0:
            raise ValueError('Number of systems cannot be zero.')
        return count

    @cached_property
    def system_properties(self):
        props = {}
        _systems_set = set(self.base_systems) | set(self.pure_component_systems or [])
        for system in _systems_set:
            base = self.pure_component_path if system in self.pure_component_systems else self.base_path
            path = os.path.join(base, system)
            props[system] = SystemProperties(path, self.ensemble)
        return props

    @cached_property
    def top_molecules(self):
        # unique molecule names present in systems
        mols_present = set()
        for system in self._systems_set:
            mols_present.update(self.system_properties[system].topology.molecules)
        return list(mols_present)
    
    @property
    def salt_pairs(self):
        """Returns a list of salt pairs."""
        return self._salt_pairs
    
    @salt_pairs.setter
    def salt_pairs(self, pairs):
        """Sets the salt pairs."""
        if not isinstance(pairs, list):
            raise TypeError(f"Expected a list of salt pairs, got {type(pairs).__name__}: {pairs}")
        if not all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs):
            raise ValueError("Each salt pair must be a tuple of two elements (cation, anion).")
        # ensure molecules in pairs are in top_molecules
        for pair in pairs:
            if not all(mol in self.top_molecules for mol in pair):
                raise ValueError(f"Salt pair {pair} contains molecules not present in top_molecules: {self.top_molecules}")
        self._salt_pairs = pairs
    
    @cached_property
    def nosalt_molecules(self):
        # get unique molecules after removing salt pairs
        _nosalt_molecules = [mol for mol in self.top_molecules if mol not in [x for pair in self.salt_pairs for x in pair]]
        return _nosalt_molecules
    
    @cached_property
    def unique_molecules(self):
        # get unique molecules after removing salt pairs
        _gm_molecules = self.nosalt_molecules.copy()
        _gm_molecules.extend(['-'.join(pair) for pair in self.salt_pairs])
        return _gm_molecules

    @property
    def n_comp(self):
        # number of molecule types present in systems
        return len(self.unique_molecules)

    def _system_total_molecules(self):
        # returns dictionary of key: systems, value: total molecules in system
        return {system: self.system_properties[system].topology.total_molecules for system in self.systems}

    def _system_molecule_counts(self):
        # returns dictionary of key: systems, value: (dict) key: molecule, value: molecule counts
        return {system: self.system_properties[system].topology.molecule_counts for system in self.systems}
    
    def _cached_lookup(self, key, compute_fn):
        if not hasattr(self, '_cache'):
            self._cache = {}
        if key not in self._cache:
            if isinstance(compute_fn, types.FunctionType):
                self._cache[key] = compute_fn()
            else:
                self._cache[key] = compute_fn
        return self._cache[key]
        
    def _pure_n_elec(self):
        """Returns a dictionary: molecule -> valence electrons."""
        key = ('_pure_n_elec')

        def compute_electron_dict():
            electron_dict = defaultdict(int)
            for system in self.systems:
                top = self.system_properties[system].topology
                for mol in top.molecules:
                    # Only set if not already set (defaultdict ensures this is safe)
                    if electron_dict[mol] == 0:
                        electron_dict[mol] = top.electron_count[mol]
            return electron_dict
                        
        return self._cached_lookup(key, compute_electron_dict)

    def _system_mol_fr(self):
        # returns dict of dict for mol fractions of each molecule in each system
        return {
            system: {mol: count / self._system_total_molecules()[system] for mol, count in mol_counts.items()}
            for system, mol_counts in self._system_molecule_counts().items()
        }

    def _system_temperatures(self, units="K"):
        key = ("_system_temperature", units)
        return self._cached_lookup(
            key, lambda: {
                system: self.system_properties[system].temperature(start_time=self.start_time, units=units)
                for system in self.systems
            }
        )

    def _system_volumes(self, units="nm^3"):
        key = ("_system_volumes", units)
        return self._cached_lookup(
            key, lambda: {
                system: self.system_properties[system].volume(start_time=self.start_time, units=units)
                for system in self.systems
            }
        )

    def _pure_molar_volumes(self, units="nm^3/molecule"):
        key = ("_pure_molar_volumes", units)
        V_unit, N_unit = units.split('/')
        return self._cached_lookup(      
            key, lambda: {
                self.system_properties[system].topology.molecules[0]: self._system_volumes(units=V_unit)[system] / self.Q_(self._system_total_molecules()[system], "molecule").to(N_unit).magnitude
                for system in self.pure_component_systems
            }
        )

    def _system_enthalpies(self, units="kJ/mol"):
        key = ("_system_enthalpies", units)
        return self._cached_lookup( 
            key, lambda: {
                system: self.system_properties[system].enthalpy(start_time=self.start_time, units=units)# / self._system_total_molecules()[system]
                for system in self.systems
            }
        )

    def _pure_enthalpies(self, units="kJ/mol"):
        key = ("_pure_enthalpies", units)
        return self._cached_lookup( 
            key, lambda: {
                self.system_properties[system].topology.molecules[0]: self._system_enthalpies(units=units)[system]
                for system in self.pure_component_systems
            }
        )
    
    def _system_ideal_mixing_enthalpy(self, units="kJ/mol"):
        key = ("_system_ideal_mixing_enthalpy", units)
        return self._cached_lookup(
            key, lambda: {
                system: sum(self._system_mol_fr().get(system, {})[mol] * self._pure_enthalpies(units=units).get(mol, 0) for mol in self._system_mol_fr().get(system, {}))
                for system in self.systems
            }
        )
    
    def _system_mixing_enthalpy(self, units="kJ/mol"):
        key = ("_system_mixing_enthalpy", units)
        return self._cached_lookup(
            key, lambda: {
                system: self._system_enthalpies(units=units).get(system, 0) - self._system_ideal_mixing_enthalpy(units=units).get(system, 0)
                for system in self.systems
            }
        )

    @cached_property
    def top_mol_fr(self):
        return np.array([
            [mfr.get(mol, 0) for mol in self.top_molecules]
            for system, mfr in self._system_mol_fr().items()
        ])
    
    @cached_property
    def mol_fr(self):
        mfr = np.zeros((self.n_sys, self.n_comp))
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
        return np.fromiter(self._system_total_molecules().values(), dtype=int)
    
    @cached_property
    def molecule_counts(self):
        return np.array([
            [counts.get(mol, 0) for mol in self.top_molecules]
            for system, counts in self._system_molecule_counts().items()
        ])
    
    def n_elec(self):
        # number of electrons
        return np.array([self._pure_n_elec()[mol] for mol in self.top_molecules])
    
    def n_elec_bar(self):
        # linear combination of number of electrons
        return self.top_mol_fr @ self.n_elec()
    
    def delta_n_elec(self):
        return self.n_elec()[:-1] - self.n_elec()[-1]

    def T(self, units="K"):
        return np.fromiter(self._system_temperatures(units=units).values(), dtype=float)
    
    def volume(self, units="nm^3"):
        return np.fromiter(self._system_volumes(units=units).values(), dtype=float)

    def rho(self, units="molecule/nm^3"):
        n_units, v_units = units.split('/')
        N = self.mol_fr * self.total_molecules[:, np.newaxis] #  calculate number of molecules
        N = self.Q_(N, "molecule").to(n_units).magnitude # convert to desired units
        V = self.volume(units=v_units)[:,np.newaxis]
        return N / V
    
    def molar_volume(self, units="nm^3/molecule"):
        return np.array([self._pure_molar_volumes(units)[mol] for mol in self.top_molecules])
    
    def delta_V(self, units="nm^3/molecule"):
        return self.molar_volume(units)[:-1] - self.molar_volume(units)[-1]
    
    def V_bar(self, units="nm^3/molecule"):
        return self.top_mol_fr @ self.molar_volume(units=units)

    def rho_ij(self, units="molecule/nm^3"):
        # shape: (n_sys, n_comp, n_comp)
        return self.rho(units=units)[:, :, np.newaxis] * self.rho(units=units)[:, np.newaxis, :]  

    def Hmix(self, units="kJ/mol"):
        return np.fromiter(self._system_mixing_enthalpy(units=units).values(), dtype=float)
    
