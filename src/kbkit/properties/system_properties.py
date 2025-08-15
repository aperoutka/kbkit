import numpy as np
import os

from .topology import TopologyParser
from .energy_reader import EnergyReader
from ..unit_registry import load_unit_registry

class SystemProperties:
    """
    Accesses system properties (both energy and topology) for a GROMACS molecular dynamics simulation. 
    Includes specific property calculations from functions in parent classes.

    Parameters
    ----------
    syspath: str
        Absolute path to system.
    ensemble: str, optional
        Ensemble for simulations. Default is '`npt`'. 

    See also
    --------
    :class:`kbkit.properties.topology.TopologyParser`: Topology parent class for molecule names, molecule numbers, and electron numbers.
    :class:`kbkit.properties.energy_reader.EnergyReader`: GROMACS properties to calculate with gmx energy.
    """
    def __init__(self, syspath, ensemble="npt"):
        self.topology = TopologyParser(syspath, ensemble)
        self.energy = EnergyReader(syspath, ensemble) # system path that contains .top and .edr files
        self.ureg = load_unit_registry()  # Load the unit registry for unit conversions

    def __getattr__(self, name):
        """
        Dynamically resolves and computes system properties as attributes.

        Parameters
        ----------
        name : str
            Name of the property to resolve.

        Returns
        -------
        function
            A function that computes the requested property with optional arguments for units, time range, 
            and output format.
        """
        # get energy attribute
        prop = self.energy._resolve_attr_key(name)         
        
        def prop_getter(time_units="ns", units=None, start_time=0, return_std=False, timeseries=False):
            # first search for heat capacity (unique case)
            if prop == "heat_capacity":
                result = self.energy.heat_capacity(
                    start_time=start_time,
                    nmol=self.topology.total_molecules,
                    units=units
                )

            # then if volume and ensemble is nvt
            elif prop == "volume" and self.energy.ensemble == "nvt":
               return self.topology.box_volume(units=units)

            # if timeseries is desired, return arrays instead of floats
            elif timeseries:
                result = self.energy.stitch_property_timeseries(
                    prop,
                    start_time=start_time,
                    time_units=time_units,
                    units=units
                )
            
            # defaults to the averaged property
            else:
                result = self.energy.average_property(
                    prop,
                    start_time=start_time,
                    units=units,
                    return_std=return_std
                )
            
            return result
        
        # special case for enthalpy
        if prop == "enthalpy":
            def enthalpy_getter(start_time=0, units=None, return_std=False):
                # get potential energy
                U = self.energy.average_property(
                    "potential",
                    start_time=start_time,
                    units="kJ/mol",
                    return_std=return_std
                )
                # get pressure
                P = self.energy.average_property(
                    "pressure",
                    start_time=start_time,
                    units="kPa",
                    return_std=return_std
                )
                # get volume, from gmx energy (npt) or .gro file (nvt)
                if self.energy.ensemble == "npt":
                    V = self.energy.average_property(
                        "volume",
                        start_time=start_time,
                        units="m^3",
                        return_std=return_std
                    )
                elif self.energy.ensemble == "nvt":
                    # For NVT, volume is not directly computed, so we use the box volume from the topology
                    V = self.topology.box_volume(units="m^3")

                H = U + P * V # enthalpy from potential energy
                H /= self.topology.total_molecules  # Convert to per molecule
                units = "kJ/mol" if units is None else units  # Default to kJ/mol if no units specified

                # convert units
                try:
                    H = self.ureg.Quantity(H, "kJ/mol").to(units).magnitude
                except Exception as e:
                    raise RuntimeError(f"Failed to convert enthalpy units: {e}") from e
                
                return float(H)
            
            return enthalpy_getter
        
        # Return the property getter function so it can accept optional units argument
        else:
            return prop_getter
    
    def get(self, name, **kwargs):
        """
        Get average property from gmx energy with automatic reading of topology information.

        Parameters
        ----------
        name: str
            Property to retrieve from gmx energy.

        Returns
        -------
        float or list[float, float]
            Scalar of average property if `return_std` option is False, else list of (average, standard deviation).
        """
        return getattr(self, name)(**kwargs)
    
    def plot(self, property_name, **kwargs):
        """Plotting function for gmx energy timeseries property.
        
        Parameters
        ----------
        property_name: str 
            Property to plot from gmx energy
            
        See also
        --------
        :meth:`kbkit.properties.energy_reader.EnergyReader.plot_property`: More details on the gmx energy plotting function.
        """
        self.energy.plot_property(
            property_name,
            **kwargs
        )

