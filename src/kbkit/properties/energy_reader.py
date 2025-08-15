import numpy as np
import glob
import os
import re
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
plt.style.use(Path(__file__).parent.parent / 'presentation.mplstyle')
from natsort import natsorted
import difflib

from ..unit_registry import load_unit_registry
from ..utils import format_unit_str, format_quantity, _find_file


class EnergyReader:

    _alias_map = {
        "enthalpy": {"enthalpy", "enth", "h", "H"},
        "temperature": {"temperature", "temp", "t"},
        "volume": {"volume", "vol", "v"},
        "heat_capacity": {"cv", "c_v", "C_v", "Cv", "cp", "c_p", "C_p", "Cp", "heat_capacity", "heat_cap"},
        "pressure": {"pressure", "pres", "p"},
        "density": {"density", "rho"},
        "potential": {"potential_energy", "potential", "pe", "U"},
        "kinetic-en": {"kinetic_energy", "kinetic", "ke"},
        "total-energy": {"total_energy", "etot", "total", "E"},
    }

    _gmx_unit_map = {
        "enthalpy": "kJ/mol",
        "temperature": "kelvin",
        "volume": "nm^3",
        "heat_capacity": "kJ/mol/K",    # For fluct_prop Cp
        "pressure": "bar",
        "density": "kg/m^3",
        "potential": "kJ/mol",
        "kinetic-en": "kJ/mol",
        "total-energy": "kJ/mol",
    }

    """
    Reads GROMACS energy (.edr) file and computes common properties via gmx energy.

    Parameters
    ----------
    syspath: str
        System path where .edr file(s) are located
    ensemble: str
        Ensemble name included in files. Default is 'npt'.
    """

    def __init__(self, syspath, ensemble="npt"):
        self.syspath = syspath
        self.ureg = load_unit_registry()
        self.Q_ = self.ureg.Quantity
        self.ensemble = ensemble.lower()

    def _resolve_attr_key(self, value, cutoff=0.6):
        """
        Resolve an attribute name to its canonical key using aliases and fuzzy matching.

        Parameters
        ----------
        value : str
            The attribute name to resolve.
        cutoff : float, optional
            Minimum similarity score to accept a match (default: 0.6).

        Returns
        -------
        str
            The canonical key corresponding to the input value.
        """
        # validate input
        try:
            value = value.lower()    
        except AttributeError:
            raise TypeError(f"Input value must be a string, got {type(value)}: {value!r}")
        
        best_match = None
        best_score = 0
        match_to_key = {}

        try:
        # Flatten all aliases to map them back to their canonical key
            for canonical_key, aliases in self._alias_map.items():
                # validate aliases type
                if not isinstance(aliases, (list, tuple, set)):
                    raise TypeError(f"Aliases for key '{canonical_key}' must be list/tuple/set, got {type(aliases)}")
                
                # get the score for the alias and update the best score and match
                for alias in aliases:
                    # check that alias is of the correct type
                    try:
                        alias = alias.lower()
                    except AttributeError:
                        raise TypeError(f"Alias must be a string, got {type(alias)}: {alias!r}")
                    match_to_key[alias] = canonical_key
                    score = difflib.SequenceMatcher(None, value, alias).ratio()
                    if score > best_score:
                        best_score = score
                        best_match = alias
        except Exception as e:
            raise RuntimeError(f"Error while resolving attribute key '{value}': {e}") from e
        
        # check that score obeys cuttoff mark
        if best_score >= cutoff:
            return match_to_key[best_match]
        else:
            raise KeyError(f"No close match found for '{value}' (best score: {best_score:.2f})")
 
    def _get_gmx_unit(self, name):
        """
        Retrieve the default GROMACS units for a given property.

        Parameters
        ----------
        name: str
            Property to return the GROMACS units for.
        
        Returns
        -------
        str
            Default GROMACS units for name.
        """
        prop = self._resolve_attr_key(name)
        try:
            return self._gmx_unit_map[prop]
        except KeyError:
            raise KeyError(f"Key '{prop}' does not exist in _gmx_unit_map: {self._gmx_unit_map.keys()}")
    
    @property
    def edr_file_list(self):
        """list: List of edr files found in syspath with ensemble in filename"""
        if not hasattr(self, '_edr_file_list'):
            # get list of .edr files
            self._edr_file_list = _find_file(suffix=".edr", ensemble=self.ensemble, syspath=self.syspath)
            # check that at least 1 files exists
            if len(self._edr_file_list) < 1:
                raise FileNotFoundError(f"No .edr files with '{self.ensemble}' exist.")
        return self._edr_file_list
    
    def available_properties(self):
        """
       Return a list of available properties in GROMACS energy file (first .edr file).

        Returns
        -------
        list of str
            Available gmx energy property options.
        """
        # get first .edr file in list
        edr_file = next(iter(self.edr_file_list))
        
        # run GMX energy command
        try:
            result = subprocess.run(
                ["gmx", "energy", "-f", edr_file],
                input="\n",
                text=True,
                capture_output=True
            )
        except FileNotFoundError:
            raise RuntimeError("GROMACS excedutable 'gmx' not found in PATH.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"gmx energy failed with exit code {e.returncode}:\n{e.stderr}")
        
        # Combine stdout and stderr in case terms are in either
        output = result.stdout + result.stderr
        lines = output.splitlines()

        # Find lines containing the selection list — between known phrases
        props_lines = []
        recording = False
        for line in lines:
            try:
                if re.match(r'^-+\s*$', line.strip()):    # line of dashes
                    if not recording:
                        recording = True
                        continue
                elif recording:
                    if line.strip() == "":
                        break
                    props_lines.append(line)
            except Exception as e:
                print(f"WARNING: skipped line due to parsing error: {line!r} ({e})")

        # Tokenize and filter
        tokens = []
        for line in props_lines:
            try:
                tokens.extend(line.strip().split())
            except Exception as e:
                print(f"WARNING: could not split line: {line!r} ({e})")

        props = [token for token in tokens if not token.isdigit()]
        if not props:
            print(f"WARNING: no properties found in '{edr_file}'. Output may have changed format.")
        return props
    

    def stitch_property_timeseries(self, name, start_time=0, time_units="ns", units=None):
        r"""
        For a given property, stitch results from several .edr files to create timeseries.

        Parameters
        ----------
        name: str
            Property to calculate with gmx energy.
        start_time: float, optional
            Time to start the timeseries evaluations at. Default 0.
        time_units: str, optional
            Units of start_time. Default 'ns'.
        units: str, optional
            Units of property result. Default is gmx units.

        Returns
        -------
        list[np.ndarray, np.ndarray]
            List of np.ndarray for time values and property values, respectfully.
        """
        # resolve attribute property
        prop = self._resolve_attr_key(name)

        all_time, all_values = [], []
        prev_end_time = 0

        for i, edr_file in enumerate(self.edr_file_list):
            filename = f"{prop}_{i}.xvg" if len(self.edr_file_list) > 1 else f"{prop}.xvg"
            output_file = os.path.join(self.syspath, filename)
            
            # run GMX energy if file does not exist
            if not os.path.exists(output_file):
                self._run_gmx_energy(edr_file, prop, output_file)

            # load data
            try:
                time, values = np.loadtxt(output_file, comments=["@", "#"], unpack=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load data from '{output_file}': {e}") from e
            
            # select start index
            try:
                start_time = float(start_time) # make sure is float
                start_idx = np.searchsorted(time, start_time)
                time, values = time[start_idx:], values[start_idx:]
            except Exception as e:
                raise RuntimeError(f"Error slicing data for {start_time=} in '{output_file}': {e}") from e

            # Handle time restarts
            try:
                if time[0] < prev_end_time:
                    offset = prev_end_time - time[0]
                    time += offset
            except Exception as e:
                raise RuntimeError(f"Error adjusting time offsets for '{output_file}': {e}") from e

            all_time.append(time)
            all_values.append(values)
            prev_end_time = time[-1]

        # concatenate arrays
        try:
            time_array = np.concatenate(all_time)
            values_array = np.concatenate(all_values)
        except Exception as e:
            raise RuntimeError(f"Error concatenating timeseries arrays: {e}") from e 

        # unit conversion
        try:
            if time_units:
                time_array = self.Q_(time_array, 'ps').to(time_units)

            gmx_unit = self._get_gmx_unit(name)
            if units and gmx_unit:
                values_array = self.Q_(values_array, gmx_unit).to(units)
            elif gmx_unit:
                values_array = self.Q_(values_array, gmx_unit)
        except Exception as e:
            raise RuntimeError(f"Error during unit conversion: {e}") from e

        return time_array, values_array

    def average_property(self, name, start_time=0, units=None, return_std=False):
        r"""
        Compute the average value of a property from stitched .edr files (:meth:`stitch_property_timeseries`).

        Parameters
        ----------
        name: str
            Property to calculate with gmx energy.
        start_time: float, optional
            Time to start the timeseries evaluations at. Default 0.
        time_units: str, optional
            Units of start_time. Default 'ns'.
        units: str, optional
            Units of property result. Default is gmx units.
        return_std: bool, optional
            Include standard deviation in output. Default False.

        Returns
        -------
        float or list[float, float]
            Scalar of average property or list of scalars (average, standard deviation) if `return_std` is True.
        """
        # load timeseries data
        time, values = self.stitch_property_timeseries(name, start_time, units=units)

        # calculate mean, standard deviation of values
        avg = values.magnitude.mean()
        std = values.magnitude.std()
        if return_std:
            return float(avg), float(std)
        else:
            return float(avg)
    
    def heat_capacity(self, nmol, units=None):
        """
        Compute the average heat capacity from a list of .edr files.

        Parameters
        ----------
        nmol : int
            Number of molecules (for _extract_heat_capacity_from_gmx).
        units : str, optional
            Desired units for heat capacity. Default is gmx units.

        Returns
        -------
        float
            Average heat capacity in the requested units.
        """
        # calculate heat capacity for each .edr file in list
        cp_vals = []
        for f in self.edr_file_list:
            try:
                cp_val = self._extract_heat_capacity_from_gmx(f, nmol)
                cp_vals.append(cp_val)
            except Exception as e:
                raise RuntimeError(f"Failed to extract heat capacity from '{f}': {e}") from e

        # Compute average safely
        try:
            cp_avg = np.mean(cp_vals)
        except TypeError as e:
            raise ValueError(f"Cannot compute mean: cp_vals contains non-numeric values: {cp_vals}. Original error: {e}") from e

        # Unit conversions
        try:
            gmx_unit = self._get_gmx_unit("heat_capacity")
            if units and gmx_unit:
                return float(self.Q_(cp_avg, gmx_unit).to(units).magnitude)
            else:
                return float(cp_avg)  
        except Exception as e:
            raise RuntimeError(f"Failed during unit conversion of heat capacity: {e}") from e

    
    def plot_property(self, property_name, start_time=0, units=None, xlim=None, ylim=None):
        """
        Plot gmx property timeseries with running average.

        Parameters
        ----------
        property_name: str
            gmx property to plot.
        start_time: float, optional
            Time to start the timeseries evaluations at. Default 0.
        units: str, optional
            Units to plot property in. Default is gmx default.
        xlim: tuple, optional
            Limits for x-axis.
        ylim: tuple, optional
            Limits for y-axis.
        """
        # load timeseries data
        time_qty, values_qty = self.stitch_property_timeseries(property_name, start_time=start_time, units=units)

        # extract arrays from pint.Quantity
        time, values = time_qty.magnitude, values_qty.magnitude

        # calculate running average
        run_avg = [np.mean(values[:i]) for i in range(values.size)]

        fig, ax = plt.subplots(1, 1, figsize=(5,4))
        ax.plot(time, values)
        ax.plot(time[:len(run_avg)], run_avg, c='k', label=format_quantity(values_qty))
        ax.set_xlabel(f'time / {format_unit_str(time_qty)}')
        ax.set_ylabel(f'{self._resolve_attr_key(property_name)} / {format_unit_str(values_qty)}')
        ax.legend()
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        plt.show()

    @staticmethod
    def _run_gmx_energy(edr_file, prop, output_file):
        """
        Runs the GROMACS 'gmx energy' command to extract a specified property from an energy file.
       
        Parameters
        ----------
        edr_file : str
            Path to the GROMACS energy (.edr) file.
        prop : str
            Name or index of the property to extract (as expected by 'gmx energy').
        output_file : str
            Path to the output file where the extracted property data will be saved.
        """
        # run gmx energy command, ensure absolute paths to avoid issues
        if not os.path.exists(edr_file):
            raise FileNotFoundError(f"EDR file not found: {edr_file}")
        
        # run GMX energy command
        try:
            result = subprocess.run(
                ["gmx", "energy", "-f", edr_file, "-o", output_file],
                input=f"{prop}\n",
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            raise RuntimeError("GROMACS excedutable 'gmx' not found in PATH.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"gmx energy failed with exit code {e.returncode}:\n{e.stderr}")

    def _extract_heat_capacity_from_gmx(self, edr_file, nmol):
        """
        Extracts the heat capacity from a GROMACS energy (.edr) file using the `gmx energy` command.
        Depending on the simulation ensemble ('npt' or 'nvt'), this method calculates either the heat capacity at constant pressure (Cp) or constant volume (Cv).
        The method runs the GROMACS energy analysis, parses the output for the relevant heat capacity value, and returns it in kJ/mol/K.
        
        Parameters
        ----------
        edr_file : str
            Path to the GROMACS energy (.edr) file.
        nmol : int
            Number of molecules in the system.
        
        Returns
        -------
        float
            Heat capacity (Cp or Cv) in kJ/mol/K.
        """

        # Determine properties and regex based on ensemble
        try:
            if self.ensemble == 'npt':
                props = "Enthalpy\nTemperature\n"
                regex = r"Heat capacity at constant pressure Cp\s+=\s+([\d\.Ee+-]+)"
            elif self.ensemble == 'nvt':
                props = "total-energy\nTemperature\n"
                regex = r"Heat capacity at constant volume Cv\s+=\s+([\d\.Ee+-]+)"
            else:
                raise ValueError(f"Unsupported ensemble: {self.ensemble}")
        except Exception as e:
            raise RuntimeError(f"Failed to set properties for ensemble '{self.ensemble}': {e}") from e


        # Run gmx energy
        try:
            result = subprocess.run(
                ["gmx", "energy", "-f", edr_file, "-nmol", str(nmol), "-fluct_props", "-driftcorr"],
                input=props,
                text=True,
                capture_output=True,
                check=True
            )
        except FileNotFoundError:
            raise RuntimeError("GROMACS executable 'gmx' not found in PATH.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"GROMACS energy command failed (exit {e.returncode}): {e.stderr}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error running gmx energy: {e}") from e

        # Parse heat capacity from output
        try:
            match = re.search(regex, result.stdout)
            if match:
                return float(match.group(1)) / 1000  # convert J/mol/K to kJ/mol/K
            else:
                print("Full GROMACS output:\n", result.stdout)
                raise ValueError("Heat capacity not found in gmx energy output.")
        except Exception as e:
            raise ValueError(f"Failed to parse heat capacity from gmx output: {e}") from e

    @classmethod
    def from_edr(cls, edr_file, **kwargs):
        """
        Initialize an instance of :class:`EnergyReader` from a single .edr file.

        Parameters
        ----------
        edr_file : str
            Path to the .edr file to be read.
        **kwargs
            Additional keyword arguments passed to the class initializer.

        Returns
        -------
        EnergyReader
            An instance of :class:`EnergyReader` initialized with the provided .edr file.
        """
        edr_dir = os.path.dirname(edr_file)
        instance = cls(edr_dir, **kwargs)
        instance._edr_file_list = [edr_file]
      
        try:
            # check that edr_file is valid
            edr_dir = os.path.dirname(edr_file)
            if not os.path.isdir(edr_dir):
                raise FileNotFoundError(f"Directory '{edr_dir}' does not exist.")

            # initialize class with edr directory
            try:
                instance = cls(edr_dir, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize class instance with directory '{edr_dir}': {e}") from e

            # validate edr_file
            if not os.path.isfile(edr_file):
                raise FileNotFoundError(f"EDR file '{edr_file}' does not exist.")
            
            # generate edr file list for obj
            instance._edr_file_list = [edr_file]

        except Exception as e:
            raise RuntimeError(f"Error creating instance from EDR file '{edr_file}': {e}") from e

        return instance

