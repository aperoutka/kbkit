from multiprocessing import Value
import re
import os
import numpy as np
from collections import defaultdict
from rdkit.Chem import GetPeriodicTable

from ..utils import _find_file
from ..unit_registry import load_unit_registry

class TopologyParser:
    """
    Extracting topology information from GROMACS .top and .gro files.

    Parameters
    ----------
    syspath: str
        Absolute system path containing .top and .gro files
    ensemble: str
        Ensemble used for molecular dynamics simulation. Options: '`npt`', '`nvt`'. Default '`npt`'.
    """
    def __init__(self, syspath, ensemble="npt"):
        self.syspath = syspath
        self.ensemble = ensemble

    def __repr__(self):
        """
        Example for a system containing ethanol (ETHOL) and H2O (TIP4P): 
            >>> obj = TopologyParser(my_system/)
            >>> print(obj)
            <TopologyParser file='my_system/topol.top' molecules={'TIP4P': 1234, 'ETHOL': 56}>
        """
        top = getattr(self, '_top_file', 'unknown.top')
        try:
            mols = getattr(self, '_molecule_counts', 'not parsed')
        except Exception:
            mols = 'unavailable'
        return f"<TopologyParser: {top} | Molecules: {mols}>"
    
    @property
    def _gro_files(self):
        # find .gro files present in syspath
        files = _find_file(suffix=".gro", ensemble=self.ensemble, syspath=self.syspath)
        return files

    @property
    def _top_file(self):
        # find .top file if present in syspath
        files = _find_file(suffix=".top", ensemble='', syspath=self.syspath)
        if not files:
            raise FileNotFoundError(f"No .top file found in path '{self.syspath}'")
        return files[0]

    def _parse_top(self):
        # reads the topology file and returns dictionary of molecule names and numbers
        molecules = {}
        in_molecules_section = False
        with open(self._top_file, 'r') as f:
            for line in f:
                # Remove comments (anything after a semicolon) and leading/trailing whitespace
                line = line.split(';')[0].strip()

                # Skip empty lines
                if not line:
                    continue

                # search for 'molecules' line
                if 'molecules' in line:
                    in_molecules_section = True
                    continue    # Move to the next line
                elif in_molecules_section and line.startswith('['):
                    # Stop parsing if we encounter another section
                    in_molecules_section = False
                    break

                # if 'molecules' found, get names & numbers
                if in_molecules_section:
                    # Split the line by spaces and tabs, filtering out empty strings
                    parts = [p.strip() for p in re.split(r'\s+', line) if p.strip()]
                    if len(parts) == 2:
                        molecule_name = parts[0]
                        try:
                            num_copies = int(parts[1])
                            molecules[molecule_name] = num_copies
                        except ValueError:
                            raise ValueError(f"Could not convert number of copies to integer for molecule '{molecule_name}'. Skipping.")

        if len(molecules) == 0:
            raise ValueError(f'Error reading top file. No molecules detected in system.')
    
        self._molecule_counts = molecules
        return self._molecule_counts

    @property
    def molecule_counts(self):
        """dict[str, int]: Dictionary of molecules present and their corresponding numbers."""
        if not hasattr(self, '_molecule_counts'):
            self._parse_top()
        # dict of molecules and their numbers
        return self._molecule_counts

    @property
    def molecules(self):
        """list: List containing names of molecules present."""
        # returns molecules names in top file
        if not hasattr(self, '_molecule_counts'):
            self._parse_top()
        return list(self._molecule_counts.keys())

    @property
    def total_molecules(self):
        """int: Total number of molecules present."""
        # total number of molecules present in system
        if not hasattr(self, '_molecule_counts'):
            self._parse_top()
        return sum(self._molecule_counts.values())
    
    def _get_atomic_number(self, atom_name):
        """Extract the atomic number"""
        # search for atomic number in atom_name str
        match = re.match(r"[A-Za-z]+", atom_name)
        if not match:
            return None

        ptable = GetPeriodicTable() # get periodic table
        symbol = match.group(0).capitalize() # capitalize the matched atom name

        # Try full 2-letter match, then fallback to 1-letter
        for key in (symbol[:2], symbol[0]):
            element_symbol = ptable.GetAtomicNumber(key) 
            if element_symbol:
                return element_symbol
        print(f"Unknown or invalid symbol '{symbol}' from atom '{atom_name}'")
    
    def _electrons_per_molecule(self):
        """
        Parse a .gro file and compute total valence electrons per residue.

        Parameters
        ----------
        gro_file : str, optional
            Path to the .gro file. If None, the first file in self._gro_files is used.

        Returns
        -------
        electron_dict : dict
            Mapping of residue names to total valence electrons in the first molecule.
        """
        # check if gro file exists
        if not self._gro_files:
            # return dict of zeros if not found---not essential for KB analysis, just used for I0 calculation
            return {mol: 0 for mol in self.molecules}

        # read .gro file
        with open(self._gro_files[0], 'r') as f:
            lines = f.readlines()

        # check if file is a valid file
        if len(lines) < 3:
            raise ValueError(f"File '{self._gro_files[0]}' is too short to be a valid .gro file.")

        # Parse number of atom
        try:
            n_atoms = int(lines[1].strip())
        except ValueError:
            raise ValueError(f"Second line of '{self._gro_files[0]}' should contain number of atoms, got: {lines[1]!r}.")
        
        if len(lines) < 2 + n_atoms:
            raise ValueError(f"The file '{self._gro_files[0]}' contains fewer atom lines ({len(lines)-2}) than expected ({n_atoms}).")
        atom_lines = lines[2:2 + n_atoms]

        # Collect unique atom names per residue from first molecule
        first_res_atoms = defaultdict(set)
        first_res_indices = {}

        for i, line in enumerate(atom_lines, start=3): # start=3 to match file line numbers
            if len(line) < 15:
                raise ValueError(f"Line {i} too short to parse residue/atom info: {line!r}")
            
            try:
                res_index = int(line[0:5]) # get residue number
            except ValueError:
                raise ValueError(f"Could not parse residue index on line {i}: {line[0:5]!r}")
            
            res_name = line[5:10].strip() # get residue name
            atom_name = line[10:15].strip() # get atom name

            # Record only atoms from the first molecule of each residue
            if res_name not in first_res_indices:
                first_res_indices[res_name] = res_index
            if res_index == first_res_indices[res_name]:
                first_res_atoms[res_name].add(atom_name)

        # Compute total valence electrons using atomic number (fallback: 0)
        electron_dict = {}
        for res_name, atom_names in first_res_atoms.items():
            total_electrons = 0
            for atom in atom_names:
                try:
                    total_electrons += self._get_atomic_number(atom)
                except Exception as e:
                    print(f"WARNING: Could not get atomic number for atom '{atom}' in residue '{res_name}': {e}")
            electron_dict[res_name] = total_electrons
        return electron_dict

    @property
    def electron_counts(self):
        """dict[str, int]: Dictionary of molecules present and their number of total electrons."""
        if not hasattr(self, '_electron_dict'):
            self._electron_dict = self._electrons_per_molecule()
        return self._electron_dict

    def box_volume(self, units=None):
        """
        Compute the average box volume from a list of .gro files.

        Parameters
        ----------
        units : str, optional
            Desired units for the volume (default: 'nm^3').

        Returns
        -------
        float
            Average volume across .gro files in the specified units.
        """
        # check if .gro file is found
        if not self._gro_files:
            raise FileNotFoundError("No .gro files available. Check that ensemble name is included and .gro files are in syspath.")        

        volume = np.zeros(len(self._gro_files))
        ureg = load_unit_registry()
        target_units = "nm^3" if units is None else units

        for i, file in enumerate(self._gro_files):
            # read .gro file
            try:
                with open(file, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                raise IOError(f"Failed to read the file '{file}': {e}")
            
            if not lines:
                raise ValueError(f"The file '{file}' is empty.")
            
            # Box vectors are expected on the last line
            box_line = lines[-1].strip().split()
            if len(box_line) < 3:
                raise ValueError(f"Last line of '{file}' does not contain enough box dimensions: {lines[-1]!r}")
            
            # convert box vectors to float
            try:
                box_dims = list(map(float, box_line[:3])) # Take only x, y, z lengths
            except ValueError as e:
                raise ValueError(f"Failed to convert box dimensions to float in '{file}': {e}")

            # Compute volume (assuming orthorhombic box)
            volume[i] = box_dims[0] * box_dims[1] * box_dims[2]
        
        # calculate the average box volume
        V = volume.mean()
        
        # convert units to target_units
        try:
            return ureg.Quantity(V, "nm^3").to(target_units).magnitude
        except Exception as e:
            raise ValueError(f"Failed to convert volume to units '{target_units}': {e}")
        
