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
        "Cp": {"cp", "c_p", "C_p", "Cp"},
        "Cv": {"cv", "c_v", "C_v", "Cv"},
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
        value = value.lower()    
        best_match = None
        best_score = 0
        match_to_key = {}
        # Flatten all aliases to map them back to their canonical key
        for canonical_key, aliases in self._alias_map.items():
            for alias in aliases:
                alias = alias.lower()
                match_to_key[alias] = canonical_key
                score = difflib.SequenceMatcher(None, value, alias).ratio()
                if score > best_score:
                    best_score = score
                    best_match = alias
        if best_score >= cutoff:
            return match_to_key[best_match]
        else:
            raise KeyError(f"No close match found for: '{value}'")
 
    def _get_gmx_unit(self, name):
        # retrieve default gromacs units
        prop = self._resolve_attr_key(name)
        return self._gmx_unit_map.get(prop)
    
    @property
    def edr_file_list(self):
        """list: List of edr files found in syspath with ensemble in filename"""
        if not hasattr(self, '_edr_file_list'):
            self._edr_file_list = _find_file(suffix=".edr", ensemble=self.ensemble, syspath=self.syspath)
        return self._edr_file_list
    
    def available_properties(self):
        """
        Print a list of avaiable properties in gmx energy for the first .edr file found in list.

        Returns
        -------
        list
            gmx property options
        """
        edr_file = next(iter(self.edr_file_list))

        result = subprocess.run(
            ["gmx", "energy", "-f", edr_file],
            input="\n",
            text=True,
            capture_output=True
        )

        # Combine stdout and stderr in case terms are in either
        output = result.stdout + result.stderr
        lines = output.splitlines()

        # Find lines containing the selection list — between known phrases
        props_lines = []
        recording = False
        for line in lines:
            if re.match(r'^-+\s*$', line.strip()):    # line of dashes
                if not recording:
                    recording = True
                    continue
            elif recording:
                if line.strip() == "":
                    break
                props_lines.append(line)

        # Tokenize and filter
        tokens = []
        for line in props_lines:
            tokens.extend(line.strip().split())
        props = [token for token in tokens if not token.isdigit()]
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
        prop = self._resolve_attr_key(name)
        all_time, all_values = [], []
        prev_end_time = 0

        for i, edr_file in enumerate(self.edr_file_list):
            filename = f"{prop}_{i}.xvg" if len(self.edr_file_list) > 1 else f"{prop}.xvg"
            output_file = os.path.join(self.syspath, filename)
            if not os.path.exists(output_file):
                self._run_gmx_energy(edr_file, prop, output_file)

            time, values = np.loadtxt(output_file, comments=["@", "#"], unpack=True)
            start_idx = np.searchsorted(time, start_time)
            time, values = time[start_idx:], values[start_idx:]

            # Detect if time has restarted (typical if time[0] < prev_end_time)
            if time[0] < prev_end_time:
                offset = prev_end_time - time[0]
                time += offset
            else:
                # No restart, if there's a gap keep it or shift to maintain monotonicity if desired
                if time[0] > prev_end_time:
                    # Optional: could shift to close gap, or keep as is
                    pass

            all_time.append(time)
            all_values.append(values)
            prev_end_time = time[-1]

        time_array = np.concatenate(all_time)
        values_array = np.concatenate(all_values)

        # unit conversion
        if time_units:
            time_array = self.Q_(time_array, 'ps').to(time_units)

        gmx_unit = self._get_gmx_unit(name)
        if units and gmx_unit:
            values_array = self.Q_(values_array, gmx_unit).to(units)
        elif gmx_unit:
            values_array = self.Q_(values_array, gmx_unit)

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
        time, values = self.stitch_property_timeseries(name, start_time, units=units)
        with np.errstate(divide='ignore', invalid='ignore'):
            avg = np.mean(values.magnitude)
            std = np.std(values.magnitude)
        if return_std:
            return float(avg), float(std)
        return float(avg)

    def to_dataframe(self, properties, start_time=0):
        """Create pandas.dataframe object from list of .edr properties for time series.
        
        Parameters
        ----------
        properties: list
            List of properties to evaluate with gmx energy
        start_time: float, optional
            Time to start the timeseries evaluations at. Default 0.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame of timeseries properties for gmx properties of interest.
        """
        properties = list(properties)
        data = {}
        for prop in properties:
            time, values_qty = self.stitch_property_timeseries(prop, start_time)
            if 'time' not in data:
                data['time'] = time
            canonical_prop = self._resolve_attr_key(prop)
            data[canonical_prop] = values_qty.magnitude
        return pd.DataFrame(data)
    
    def heat_capacity(self, nmol, units=None):
        """
        Calculate the heat capacity from gmx energy.

        Parameters
        ----------
        nmol: int
            Number of total molecules present in simulation
        units: str, optional
            Units for heat capacity result
        
        Returns
        -------
        float
            Heat capacity scalar from simulation
        """
        cp_vals = [self._extract_heat_capacity_from_gmx(f, nmol) for f in self.edr_file_list]
        with np.errstate(divide='ignore', invalid='ignore'):
            cp_avg = np.mean(cp_vals)
        # unit conversions
        gmx_unit = self._get_gmx_unit("heat_capacity")
        if units and gmx_unit:
            cp_qty = self.Q_(cp_avg, gmx_unit).to(units)
        elif gmx_unit:
            cp_qty = self.Q_(cp_avg, gmx_unit)
        return float(cp_qty)

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
        time_qty, values_qty = self.stitch_property_timeseries(property_name, start_time=start_time, units=units)
        time, values = time_qty.magnitude, values_qty.magnitude
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
        # run gmx energy command, ensure absolute paths to avoid issues
        if not os.path.exists(edr_file):
            raise FileNotFoundError(f"EDR file not found: {edr_file}")
        result = subprocess.run(
            ["gmx", "energy", "-f", edr_file, "-o", output_file],
            input=f"{prop}\n",
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode != 0:
            raise RuntimeError(f"gmx energy command failed with exit code {result.returncode}")

    def _extract_heat_capacity_from_gmx(self, edr_file, nmol):
        # calculate heat capacity at constant pressure from gmx energy
        if not os.path.exists(edr_file):
            raise FileNotFoundError(f"EDR file not found: {edr_file}")
        
        if self.ensemble == 'npt':
            props = "Enthalpy\nTemperature\n"
            match = re.search(r"Heat capacity at constant pressure Cp\s+=\s+([\d\.Ee+-]+)", result.stdout)
        elif self.ensemble == 'nvt':
            props = "total-energy\nTemperature\n"
            match = re.search(r"Heat capacity at constant volume Cv\s+=\s+([\d\.Ee+-]+)", result.stdout)

        result = subprocess.run(
            ["gmx", "energy", "-f", edr_file, "-nmol", str(nmol), "-fluct_props", "-driftcorr"],
            input=props,
            text=True,
            capture_output=True
        )

        if result.returncode != 0:
            raise RuntimeError("GROMACS energy command failed")

        # Parse heat-capacity from the correct line
        if match:
            return float(match.group(1)) / 1000 # convert to kJ/mol/K from J/mol/K
        else:
            print("Full GROMACS output:\n", result.stdout)
            raise ValueError("Heat capacity not found in gmx energy output")

    @classmethod
    def from_edr(cls, edr_file, **kwargs):
        """Initialize :class:`EnergyReader` from a single .edr file path."""
        edr_dir = os.path.dirname(edr_file)
        instance = cls(edr_dir, **kwargs)
        instance._edr_file_list = [edr_file]
        return instance

