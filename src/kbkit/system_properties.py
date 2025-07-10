import numpy as np
import os

from .properties.topology import TopologyParser
from .properties.energy_reader import EnergyReader
from .unit_registry import load_unit_registry

class SystemProperties:
    # class to hold system properties for a molecular dynamics simulation.
    # and specific property calculation from functions in parent classes
    def __init__(self, syspath, ensemble="npt"):
        self.topology = TopologyParser(syspath, ensemble)
        self.energy = EnergyReader(syspath, ensemble) # system path that contains .top and .edr files
        self.ureg = load_unit_registry()  # Load the unit registry for unit conversions

    def __getattr__(self, name):
        # Allow optional units keyword argument to override defaults
        prop = self.energy._resolve_attr_key(name)         
        
        def prop_getter(time_units="ns", units=None, start_time=0, return_std=False, timeseries=False):
            # Compute the property and store it
            if any(x in prop for x in ["Cp", "Cv"]):
                result = self.energy.heat_capacity(
                    start_time=start_time,
                    nmol=self.topology.total_molecules,
                    units=units
                )
            elif prop == "volume" and self.energy.ensemble == "nvt":
               return self.topology.box_volume(units=units)

            elif timeseries:
                result = self.energy.stitch_property_timeseries(
                    prop,
                    start_time=start_time,
                    time_units=time_units,
                    units=units
                )
            else:
                result = self.energy.average_property(
                    prop,
                    start_time=start_time,
                    units=units,
                    return_std=return_std
                )
            
            return result
        
        if prop == "enthalpy":
            # Special case for enthalpy, which is computed differently
            def enthalpy_getter(start_time=0, units=None, return_std=False):
                U = self.energy.average_property(
                    "potential",
                    start_time=start_time,
                    units="kJ/mol",
                    return_std=return_std
                )
                P = self.energy.average_property(
                    "pressure",
                    start_time=start_time,
                    units="kPa",
                    return_std=return_std
                )
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
                # Enthalpy H = U + PV
                H = U + P * V
                H /= self.topology.total_molecules  # Convert to per molecule
                units = "kJ/mol" if units is None else units  # Default to kJ/mol if no units specified
                H = self.ureg.Quantity(H, "kJ/mol").to(units).magnitude  # Convert to requested units
                if return_std:
                    H_std = np.std(H)
                    return float(H), float(H_std)
                return float(H)
            return enthalpy_getter
        else:
            # Return the property getter function so it can accept optional units argument
            return prop_getter
    
    def get(self, name, **kwargs):
        return getattr(self, name)(**kwargs)
    
    def plot(self, property_name, **kwargs):
        self.energy.plot_property(
            property_name,
            **kwargs
        )

