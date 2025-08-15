import os
import numpy as np
from scipy.integrate import cumulative_trapezoid
from uncertainties.umath import *
import matplotlib.pyplot as plt
from pathlib import Path

from ..properties.system_properties import SystemProperties
from ..unit_registry import load_unit_registry
from .rdf import RDF


class KBI:
    """
    Class to compute the Kirkwood-Buff Integrals (KBI) from RDF data.
    
    Parameters
    ----------
    rdf_file : str
        Path to the RDF file containing radial distances and corresponding g(r) values.
    sys_path : str, optional
        Path to the system directory containing topology files. If not provided, it will be inferred from the RDF file path.
    ensemble : str, optional
        Ensemble type for the system properties. Default is 'npt'.
    """
    def __init__(self, rdf_file, sys_path=None, ensemble='npt'):
        self.rdf = RDF(rdf_file)  
        sys_path = sys_path if sys_path is not None else self._syspath()
        self.system_properties = SystemProperties(sys_path, ensemble)

    def _syspath(self):
        """
        Searches for a topology (.top) file in the directory of the RDF file and its parent directory.

        Returns
        -------
        str
            The path to the directory containing a .top file.
        """
        # create path obj to rdf file
        rdf_path = Path(self.rdf.rdf_file)
        # check that path exists
        if not rdf_path.exists():
            raise FileNotFoundError(f"RDF file does not exist: {rdf_path}")
        
        # directories to check: RDF dir and its parent
        for directory in [rdf_path.parent, rdf_path.parent.parent]:
            # search for .top file
            try:
                top_files = list(directory.glob("*.top"))
            except PermissionError as e:
                raise PermissionError(f"Permission denied when accessing '{directory}': {e}") from e
            except OSError as e:
                raise RuntimeError(f"Error accessing '{directory}': {e}") from e 
            
            if top_files:
                # return first match
                return str(top_files[0].parent)
            
        raise FileNotFoundError(f"Topology (.top) file not found in '{rdf_path.parent}' or '{rdf_path.parent.parent}'")

    
    def box_vol(self):
        """float: Volume of the system box in nm^3."""
        return self.system_properties.volume(units="nm^3")
    
    def rdf_molecules(self):
        """Get the molecules corresponding to the RDF file from the system topology.
        
        Returns
        -------
        list
            List of molecule IDs corresponding to the RDF file.
        """
        # extract molecules from file name and topology information
        rdf_mols = RDF.extract_mols(self.rdf.rdf_file, self.system_properties.topology.molecules)
        # check length of molecules found --- must be two for rdfs
        if len(rdf_mols) != 2:
            raise ValueError('Number of molecules corresponding to ID in .top file is not 2!')
        return rdf_mols
       
    def kd(self):
        """
        Get the Kronecker delta, determine if molecules :math:`i,j` are the same.

        Returns
        -------
        int
            Kronecker delta between molecules in RDF.
        """
        return int(self.rdf_molecules()[0] == self.rdf_molecules()[1])

    def Nj(self):
        """
        Returns
        -------
        int
            Number of molecule :math:`j` in the system.
        """
        return self.system_properties.topology.molecule_counts[self.rdf_molecules()[1]]

    def g_gv(self):
        r"""       
        This method computes the corrected pair distribution function, accounting for finite-size effects in the simulation box, based on the approach by `Ganguly and Van der Vegt (2013) <https://doi.org/10.1021/ct301017q>`_.

        Returns
        -------
        np.ndarray
            Corrected g(r) values as a numpy array corresponding to distances `r` from the RDF.

        Notes
        -----
        The correction is calculated as

        .. math::
            v_r = 1 - \frac{\frac{4}{3} \pi r^3}{V}

        .. math::
            \rho_j = \frac{N_j}{V}

        .. math::
            \Delta N_j = \int_0^r 4 \pi r^2 \rho_j \bigl(g(r) - 1 \bigr) \, dr

        .. math::
            g_{GV}(r) = g(r) \cdot \frac{N_j v_r}{N_j v_r - \Delta N_j - \delta_{ij}}


        where:
         - :math:`r` is the distance
         - :math:`V` is the box volume
         - :math:`N_j` is the number of particles of type \( j \)
         - :math:`g(r)` is the raw radial distribution function
         - :math:`\delta_{ij}` is a kronecker delta
        
        .. note::
            The cumulative integral :math:`\Delta N_j` is approximated numerically using the trapezoidal rule.
        """
        # calculate the reduced volume
        vr = 1 - ((4/3) * np.pi * self.rdf.r**3 / self.box_vol())
        
        # get the number density for molecule j
        rho_j = self.Nj() / self.box_vol()
        
        # function to integrate over
        f = 4. * np.pi * self.rdf.r**2 * rho_j * (self.rdf.g - 1)
        Delta_Nj = cumulative_trapezoid(f, x=self.rdf.r, dx=self.rdf.r[1]-self.rdf.r[0])
        Delta_Nj = np.append(Delta_Nj, Delta_Nj[-1])
       
        # correct g(r) with GV correction
        g_gv = self.rdf.g * self.Nj() * vr / (self.Nj() * vr - Delta_Nj - self.kd())
        return g_gv
    
    def window(self):
        r"""
        This function applies a cubic correction (or window weight) to the radial distribution function, which is useful for ensuring that the integral converges properly at larger distances, based on the method described by `Krüger et al. (2013) <https://doi.org/10.1021/jz301992u>`_.

        Returns
        -------
        np.ndarray
            Windowed weight for the RDF

        Notes
        -----
        The windowed weight is defined as:

        .. math::
            w(r) = 4 \pi r^2 \left(1 - \left(\frac{r}{r_{max}}\right)^3\right)

        where: 
            - :math:`r` is the radial distance
            - :math:`r_{max}` is the maximum radial distance in the RDF
        """
        return 4 * np.pi * self.rdf.r**2 * (1 - (self.rdf.r/self.rdf.rmax)**3)
   
    def h(self):
        r"""
        Calculates the correlation function h(r) from the corrected g(r) values.

        Returns
        -------
        np.ndarray
            Correlation function h(r) as a numpy array.

        Notes
        -----
        The correlation function is defined as: 

        .. math::
            h(r) = g_{GV}(r) - 1

        """
        return self.g_gv() - 1
       
    def rkbi(self):
        r"""
        Computes the Kirkwood-Buff Integral (KBI) as a function of radial distance between molecules :math:`i` and :math:`j`.

        Returns
        -------
        np.ndarray
            KBI values as a numpy array corresponding to distances :math:`r` from the RDF.

        Notes
        -----
        The KBI is computed using the formula:

        .. math::
            G_{ij}(r) = \int_0^r h(r) w(r) dr

        where:
            - :math:`h(r)` is the correlation function
            - :math:`w(r)` is the window function
            - :math:`r` is the radial distance

        .. note::
            The integration is performed using the trapezoidal rule.
        """
        return cumulative_trapezoid(self.window() * self.h(), self.rdf.r, initial=0)

    def lambda_ratio(self):
        r"""
        Calculates the length ratio (:math::`\lambda`) of the system based on the radial distances and the box volume.
        
        Returns
        -------
        np.ndarray
            Length ratio as a numpy array corresponding to distances :math:`r` from the RDF.

        Notes
        -----
        The length ratio is defined as:

        .. math::
            \lambda = \left(\frac{\frac{4}{3} \pi r^3}{V}\right)^{1/3}
            
        where:
            - :math:`r` is the radial distance
            - :math:`V` is the box volume
        """
        Vr = (4/3) * np.pi * self.rdf.r**3 / self.box_vol()
        return Vr ** (1/3)

    def fit_kbi_inf(self):
        r"""
        Computes the KBI at infinite distance by fitting a linear model to the product of the length ratio and the KBI values.
        
        Returns
        -------
        tuple
            Tuple containing the slope and intercept of the linear fit, which represents the KBI at infinite distance.

            
        .. note::
            The KBI at infinite distance is estimated by fitting a linear model to the product of the length ratio and the KBI values, using only the radial distances that are within the specified range (rmin to rmax).
        """
        # get x and y values to fit thermodynamic correction
        l = self.lambda_ratio() # characteristic length
        l_kbi = l * self.rkbi() # length x KBI (r)

        # apply r_mask to values for extrapolation 
        l_fit = l[self.rdf.r_mask]
        l_kbi_fit = l_kbi[self.rdf.r_mask]

        # fit linear regression to masked values
        fit_params = np.polyfit(l_fit, l_kbi_fit, 1)
        return fit_params # return fit 
    
    def integrate(self):
        """
        Returns
        -------
        float
            KBI in the thermodynamic limit, which is the slope of the linear fit to the product
            of the length ratio and the KBI values.
        """
        return self.fit_kbi_inf()[0]

    def plot(self, save_dir=None):
        """Plots subplots of the RDF and the running KBI including the fit to thermodynamic limit.
        
        Parameters
        ----------
        save_dir : str, optional
            Directory to save the plot. If not provided, the plot will be displayed but not saved
        """
        fig, ax = plt.subplots(1, 2, figsize=(9,4))
        rkbi = self.rkbi()
        ax[0].plot(self.rdf.r, rkbi)
        ax[0].set_xlabel('r / nm')
        ax[0].set_ylabel('G$_{ij}$ / nm$^3$')
        l = self.lambda_ratio()
        l_kbi = l * rkbi
        ax[1].plot(l, l_kbi)
        fit_params = self.fit_kbi_inf()
        l_fit = l[self.rdf.r_mask]
        y_hat = np.polyval(fit_params, l_fit)
        ax[1].plot(l_fit, y_hat, ls='--', c='k', label=f'KBI: {fit_params[0]:.2g} nm$^3$')
        ax[1].set_xlabel('$\lambda$')
        ax[1].set_ylabel('$\lambda$ G$_{ij}$ / nm$^3$')
        fig.suptitle(f'KBI Analysis for system: {os.path.basename(self.rdf._syspath())} {self.rdf.rdf_molecules()[0]}-{self.rdf.rdf_molecules()[1]}')
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, self.rdf.rdf_file[:-4] + '.png'))
        plt.show()

    @staticmethod
    def to_cm3_mol(value_nm3):
        """Converts a value from nm^3/molecule to cm^3/mol.
        
        Parameters
        ----------
        value_nm3 : float
            Value in nm^3/molecule to be converted.
            
        Returns 
        -------
        value_cm3_mol : float
            Converted value in cm^3/mol.
        """
        # setup unit registry
        ureg = load_unit_registry()
        Q_ = ureg.Quantity
        # convert from nm3/molecule -> cm3/mol
        value_cm3_mol = Q_(value_nm3, "nm^3/molecule").to("cm^3/mol")
        return value_cm3_mol.magnitude

