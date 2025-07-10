import os
import numpy as np
from scipy.integrate import cumulative_trapezoid
from uncertainties.umath import *
import matplotlib.pyplot as plt

from kbkit.src.kbkit.system_properties import SystemProperties

from ..unit_registry import load_unit_registry
from .rdf import RDF


class KBI:
    def __init__(self, rdf_file, sys_path=None, ensemble='npt'):
        self.rdf = RDF(rdf_file)  
        sys_path = sys_path if sys_path is not None else self._syspath()
        self.system_properties = SystemProperties(sys_path, ensemble)

    def _syspath(self):
        for directory in [os.path.dirname(self.rdf.rdf_file),
                        os.path.dirname(os.path.dirname(self.rdf.rdf_file))]:
            for f in os.listdir(directory):
                if f.strip().endswith('.top'):
                    return directory
        raise FileNotFoundError("Topology (.top) file not found in system.")
    
    def box_vol(self):
        return self.system_properties.volume(units="nm^3")
    
    def rdf_molecules(self):
        rdf_mols = RDF.extract_mols(self.rdf.rdf_file, self.system_properties.topology.molecules)
        if len(rdf_mols) != 2:
            raise ValueError('Number of molecules corresponding to ID in .top file is not 2!')
        return rdf_mols
       
    def kd(self):
        return int(self.rdf_molecules()[0] == self.rdf_molecules()[1])

    def Nj(self):
        return self.system_properties.topology.molecule_counts[self.rdf_molecules()[1]]

    def g_gv(self):
        """correct g(r) for excess/depletion with Ganguly-van der Vegt"""
        vr = 1 - ((4/3) * np.pi * self.rdf.r**3 / self.box_vol())
        rho_j = self.Nj() / self.box_vol()
        f = 4. * np.pi * self.rdf.r**2 * rho_j * (self.rdf.g - 1)
        Delta_Nj = cumulative_trapezoid(f, x=self.rdf.r, dx=self.rdf.r[1]-self.rdf.r[0])
        Delta_Nj = np.append(Delta_Nj, Delta_Nj[-1])
        g_gv = self.rdf.g * self.Nj() * vr / (self.Nj() * vr - Delta_Nj - self.kd())
        return g_gv
    
    def window(self):
        """define windowed weight -- from Kruger (2013)"""
        return 4 * np.pi * self.rdf.r**2 * (1 - (self.rdf.r/self.rdf.rmax)**3)
   
    def h(self):
        """correction to correlation function"""
        # correct excess/depletion with gv (g(r)) correction
        return self.g_gv() - 1

    def integrate(self):
        return self._compute_kbi_inf()[0]
       
    def _compute_rkbi(self):
        """Integrate correlation function to get KBI for a given RDF"""
        rkbi = cumulative_trapezoid(self.window() * self.h(), self.rdf.r, initial=0)
        return rkbi

    def _lambda_ratio(self):
        Vr = (4/3) * np.pi * self.rdf.r**3 / self.box_vol()
        return Vr ** (1/3)

    def _compute_kbi_inf(self):
        """extrapolate kbi to thermodynamic limit"""
        l = self._lambda_ratio()
        l_kbi = l * self._compute_rkbi()
        l_fit = l[self.rdf.r_mask]
        l_kbi_fit = l_kbi[self.rdf.r_mask]
        fit_params = np.polyfit(l_fit, l_kbi_fit, 1)
        return fit_params

    def plot(self, save_dir=None):
        fig, ax = plt.subplots(1, 2, figsize=(9,4))
        rkbi = self._compute_rkbi()
        ax[0].plot(self.rdf.r, rkbi)
        ax[0].set_xlabel('r / nm')
        ax[0].set_ylabel('G$_{ij}$ / nm$^3$')
        l = self._lambda_ratio()
        l_kbi = l * rkbi
        ax[1].plot(l, l_kbi)
        fit_params = self._compute_kbi_inf()
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
        # setup unit registry
        ureg = load_unit_registry()
        Q_ = ureg.Quantity
        # convert from nm3/molecule -> cm3/mol
        value_cm3_mol = Q_(value_nm3, "nm^3/molecule").to("cm^3/mol")
        return value_cm3_mol.magnitude

