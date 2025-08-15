import numpy as np
import os
import copy
from scipy.integrate import cumulative_trapezoid
from scipy import constants
from functools import partial
from itertools import product
from pathlib import Path

from .rdf import RDF
from .kbi import KBI
from .system_set import SystemSet

class KBThermo(SystemSet):
    # for applying Kirkwood-Buff (KB) theory to calculate thermodynamic properties for entire systems.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _top_mol_idx(self, mol):
        # get index of mol in topology molecules list
        if mol not in self.top_molecules:
            raise ValueError(f"Molecule {mol} not in topology molecules. Topology molecules: {self.top_molecules}")
        return list(self.top_molecules).index(mol)

    def _mol_idx(self, mol):
        # get index of mol in unique molecules list
        if mol not in self.unique_molecules:
            raise ValueError(f"Molecule {mol} not in topology molecules. Unique molecules: {self.unique_molecules}")
        return list(self.unique_molecules).index(mol)

    def calculate_kbis(self):
        """
        Get Kirkwood-Buff integral (KBI) matrix, **G**, for all systems and all pairs of molecules.

        Returns
        -------
        np.ndarray
            A 3D matrix of Kirkwood-Buff integrals with shape ``(n_sys, n_mols, n_mols)``,
            where:

            - ``n_sys`` — number of systems
            - ``n_mols`` — number of unique molecules

        Notes
        -----
        For each system, element :math:`G_{ij}` of matrix **G**, is the KBI for a pair of molecules :math:`i, j`
        and computed as:

        .. math::

            G_{ij} = 4 \\pi \\, \\int_0^{\\infty} \\, (g_{ij}(r) - 1) \\, r^2 \\, dr

        where, :math:`g_{ij}(r)` is the RDF for the pair.

        The algorithm:
            1. Iterates through each system.
            2. Checks if the RDF directory exists; skips systems without RDF data.
            3. Reads RDF files for each molecular pair.
            4. Integrates RDF data to compute :math:`G_{ij}`.
            5. Stores results in a symmetric KBI matrix for the system.

        If an RDF directory is missing, the corresponding system's values remain NaN.

        See Also
        --------
        :class:`kbkit.kb.rdf.RDF` : Parses RDF files.
        :class:`kbkit.kb.kbi.KBI` : Performs the RDF integration to compute KBIs and apply finite-size corrections.
        """
        if '_kbis' not in self.__dict__:
            self._kbi_mat = np.full((self.n_sys, len(self.top_molecules), len(self.top_molecules)), fill_value=np.nan)
            
            # iterate through all systems
            for s, sys in enumerate(self.systems):
                
                # if rdf dir not in system, skip
                rdf_full_path = os.path.join(self.base_path, sys, self.rdf_dir)
                if not Path(rdf_full_path).exists():
                    continue
                
                # read all rdf_files
                for rdf_file in os.listdir(rdf_full_path):
                    rdf_file_path = os.path.join(rdf_full_path, rdf_file)
                    rdf_mols = RDF.extract_mols(rdf_file_path, self.top_molecules)
                    i, j = [self._top_mol_idx(mol) for mol in rdf_mols]
                    
                    # integrate rdf --> kbi calc
                    integrator = KBI(rdf_file_path)
                    kbi = integrator.integrate()
                    self._update_kbi_dict(system=sys, rdf_mols=rdf_mols, integrator=integrator)
                    
                    # add to matrix
                    self._kbis[s, i, j] = kbi
                    self._kbis[s, j, i] = kbi

        return self._kbis
    
    def kbi_dict(self):
        r"""
        Get a dictionary of KBI and RDF properties for each system and molecular pair.

        Returns
        -------
        dict[str, dict[str, float or numpy.ndarray]]
            A nested dictionary mapping systems and molecule pairs to RDF and KBI properties.
            Outer keys are systems, inner keys are molecule pairs, and values are either scalars(:class:`float`) or arrays (:class:`np.ndarray`).
            
        Notes
        -----
        The inner keys are defined as follows:
            - '`r`': Radial distance array from RDF.
            - '`g`': RDF values for the pair.
            - '`rkbi`': KBI value for the pair.
            - '`lambda`': Lambda ratio used in KBI calculation.
            - '`lambda_kbi`': KBI value adjusted by lambda ratio.
            - '`lambda_fit`': Lambda ratio for the fitted RDF.
            - '`lambda_kbi_fit`': KBI value adjusted by fitted lambda ratio.
            - '`kbi_inf`': Infinite dilution KBI value.
        """
        # returns dictionary of kbi / rdf properties by system and pair molecular interaction
        if not hasattr(self, '_kbi_dict'):
            self.kbi_mat()
        return self._kbi_dict
    
    def _update_kbi_dict(self, system, rdf_mols, integrator):
        # add kbi/rdf properties to dictionary; sorted by system / rdf
        if not hasattr(self, '_kbi_dict'):
            self._kbi_dict = {}

        self._kbi_dict.setdefault(system, {}).update({
            '-'.join(rdf_mols): {
                'r': integrator.rdf.r,
                'g': integrator.rdf.g,
                'rkbi': (rkbi := integrator.rkbi()),
                'lambda': (lam := integrator._lambda_ratio()),
                'lambda_kbi': lam * rkbi,
                'lambda_fit': lam[integrator.rdf.r_mask],
                'lambda_kbi_fit': np.polyval(integrator._compute_kbi_inf(), lam[integrator.rdf.r_mask]),
                'kbi_inf': integrator.integrate(),
            }
        })

    def electrolyte_kbi_correction(self, kbi_matrix):
        r"""
        Apply electrolyte correction to the input KBI matrix.
        
        This method modifies the KBI matrix to account for salt-salt and salt-other interactions
        by adding additional rows and columns for salt pairs. It calculates the KBI for salt-salt interactions
        based on the mole fractions of the salt components and their interactions with other molecules.

        Parameters
        ----------
        kbi_matrix : np.ndarray
            A 3D matrix representing the original KBI matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.
        
        Returns
        -------
        np.ndarray
            A 3D matrix representing the modified KBI matrix with additional rows and columns for salt pairs.
            
        Notes
        -----
        - If no salt pairs are defined, it returns the original KBI matrix.
        - The salt pairs are defined in ``KBThermo.salt_pairs``, which should be a list of tuples containing the names of the salt components.
        
        This method calculates the KBI matrix (**G**) for systems with salts for salt-salt interactions (:math:`G_{ss}`) and salt-other interactions (:math:`G_{si}`) as follows:

        .. math::
            G_{ss} = x_c^2 G_{cc} + x_a^2 G_{aa} + x_c x_a (G_{ca} + G_{ac})

        .. math::
            G_{si} = x_c G_{ic} + x_a G_{ia}

        .. math::
            x_c = \frac{N_c}{N_c + N_a}

        .. math::
            x_a = \frac{N_a}{N_c + N_a}
        
        where:
            - :math:`G_{ss}` is the KBI for salt-salt interactions.
            - :math:`G_{si}` is the KBI for salt-other interactions.
            - :math:`x_c` and :math:`x_a` are the mole fractions of the salt components.
            - :math:`N_c` and :math:`N_a` are the counts of the salt components in the system.
            - :math:`G_{cc}`, :math:`G_{aa}`, and :math:`G_{ca}` are the KBIs for the respective pairs of molecules.

        """
        # if no salt pairs detected return original matrix
        if len(self.salt_pairs) == 0:
            return kbi_matrix

        # create new kbi-matrix
        adj = len(self.salt_pairs)-len(self.top_molecules)
        kbi_el = np.full((self.n_sys, self.n_comp+adj, self.n_comp+adj), fill_value=np.nan)

        for i, (c, a) in enumerate(self.salt_pairs):
            # get index of anion and cation in topology molecules
            cj = self.top_molecules.index(c)
            aj = self.top_molecules.index(a)

            # mol fraction of anion/cation in anion-cation pair
            xc = self.molecule_counts[:,cj]/(self.molecule_counts[:,cj]+self.molecule_counts[:,aj])
            xa = self.molecule_counts[:,aj]/(self.molecule_counts[:,cj]+self.molecule_counts[:,aj])

            # for salt-salt interactions add to kbi-matrix
            try:
                sj = self.gm_molecules.index('-'.join([c,a]))
            except:
                sj = self.gm_molecules.index('-'.join([a,c]))
            # calculate electrolyte KBI for salt-salt pairs
            kbi_el[sj, sj] = xc**2 * kbi_matrix[cj, cj] + xa**2 * kbi_matrix[aj, aj] + xc*xa * (kbi_matrix[cj, aj] + kbi_matrix[aj, cj])

            # for salt other interactions
            for m1, mol1 in enumerate(self.nosalt_molecules):
                m1j = self.top_molecules.index(mol1)
                for m2, mol2 in enumerate(self.nosalt_molecules):
                    m2j = self.top_molecules.index(mol2)
                    kbi_el[m1, m2] = kbi_matrix[m1j, m2j]
                # adjusted KBI for mol-salt interactions
                kbi_el[m1, sj] = xc * kbi_matrix[m1, cj] + xa * kbi_matrix[m1, aj]
                kbi_el[sj, m1] = xc * kbi_matrix[cj, m1] + xa * kbi_matrix[aj, m1]

        return kbi_el
    
    def kbi_mat(self):
        """
        Get the KBI matrix (**G**) with electrolyte corrections applied if salt pairs are defined.

        Returns
        -------
        np.ndarray
            A 3D matrix representing the KBI matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of components,
            including any additional salt pairs if defined.        
        """
        if '_kbi_mat' not in self.__dict__:
            kbi_matrix = self.calculate_kbis()
            self._kbi_mat = self._electrolyte_kbi_correction(kbi_matrix=kbi_matrix.copy())
        return self._kbi_mat
    
    def kd(self):
        """
        Get the Kronecker delta between pairs of unique molecules. 
        
        Returns
        -------
        np.ndarray
            A 2D array representing the Kronecker deltas with shape ``(n_comp, n_comp)``,
            where ``n_comp`` is the number of unique components.
        """
        return np.eye(self.n_comp)
    
    def B_mat(self):
        r"""
        Construct a symmetric matrix **B** for each system based on the number densities and KBIs.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Notes
        -----
        Elements of **B** are calculated for molecules :math:`i,j`, using the formula:

        .. math::
            B_{ij} = \rho_{ij} G_{ij} + \rho_i \delta_{i,j}

        where:
            - :math:`\rho_{ij}` is the pairwise number density of molecules in each system.
            - :math:`G_{ij}` is the KBI for the pair of molecules.
            - :math:`\rho_i` is the number density of molecule :math:`i`.
            - :math:`\delta_{i,j}` is the Kronecker delta for molecules :math:`i,j`.
        """
        if '_B_mat' not in self.__dict__:
            self._B_mat = self.rho_ij(units="molecule/nm^3") * self.kbi_mat() + self.rho(units="molecule/nm^3")[:,:,np.newaxis] * self.kd()[np.newaxis,:,:]
        return self._B_mat

    @property
    def _B_inv(self):
        """np.ndarray: Inverse of the B matrix."""
        return np.linalg.inv(self.B_mat())
    
    @property
    def _B_det(self):
        """np.ndarray: Determinant of the B matrix."""
        return np.linalg.det(self.B_mat())

    def B_cofactors(self):
        r"""
        Get the cofactors of **B** for each system.
        
        Returns
        -------
        np.ndarray
            A 3D matrix representing the cofactors of **B** with shape ``(n_sys, n_comp, n_comp)``,
        
        Notes
        -----
        The cofactors of **B**, :math:`Cof(\mathbf{B})`, are calculated as:
        
        .. math::
            Cof(\mathbf{B}) = |\mathbf{B}| \cdot \mathbf{B}^{-1}
        
        where:
            - :math:`|\mathbf{B}|` is the determinant of **B**
            - :math:`\mathbf{B}^{-1}` is the inverse of **B**
        """
        if '_B_cofactors' not in self.__dict__:
            self._B_cofactors = self._B_det[:,np.newaxis,np.newaxis] * self._B_inv
        return self._B_cofactors
    
    def A_mat(self):
        r"""
        Construct a symmetric matrix **A** for each system from compositions and **G**

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Notes
        -----
        Elements of **A** are calculated for molecules :math:`i,j`, using the formula:

        .. math::
            A_{ij} = \rho x_i x_j G_{ij} + x_i \delta_{i,j}

        where:
            - :math:`\rho` is the average mixture density.
            - :math:`G_{ij}` is the KBI for the pair of molecules.
            - :math:`x_i` is the mol fraction of molecule :math:`i`.
            - :math:`\delta_{i,j}` is the Kronecker delta for molecules :math:`i,j`.        
        """
        if '_A_mat' not in self.__dict__:
            self._A_mat = (1/self.V_bar(units="nm^3/molecule"))[:,np.newaxis,np.newaxis] * self.mol_fr[:,:,np.newaxis] * self.mol_fr[:,np.newaxis,:] * self.kbi_mat() + self.mol_fr[:,:,np.newaxis] * self.kd()[np.newaxis,:,:]
        return self._A_mat

    def isothermal_compressability(self, units="kJ/mol"):
        r"""
        Calculates the isothermal compressability, :math:`\kappa`, for each system.

        
        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            Isothermal compressability values for each system, with shape ``(n_sys)``

        Notes
        -----
        Isothermal compressability (:math:`\kappa`) is calculated by: 

        .. math::
            \kappa RT = \sum_{j=1}^n V_j A_{ij}^{-1}

        where:
            - :math:`V_j` is the molar volume of molecule :math:`j`
            - :math:`A_{ij}^{-1}` is the inverse of **A** for molecules :math:`i,j`

        """
        R = self.ureg.R.to(units + "/K").magnitude # gas constant
        kT = (1 / (R * self.T())) * (self.molar_volume()[np.newaxis,:]/self.A_mat()[:,0,:]).sum(axis=1) # isothermal compressability
        return self.Q_(kT, units=f"{units.split('/')[1]}/{units.split('/')[0]} * nm^3/molecule").to("1/kPa").magnitude

    def dmu_dN(self, units="kJ/mol"):
        r"""
        The derivative of the chemical potential of molecule :math:`i` with respect to the number of molecules of molecule :math:`j`.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 3D matrix of shape ``(n_sys, n_comp, n_comp)``

        Notes
        -----
        Derivative of chemical potential with respect to molecule number (:math:`\frac{\partial \mu_i}{\partial n_j}`) is calculated as follows:

        .. math::
           \frac{\partial \mu_i}{\partial n_j} = \frac{k_bT}{\left<V\right> |\mathbf{B}|}\left(\frac{\sum_{a=1}^n\sum_{b=1}^n \rho_a\rho_b\left|B^{ij}B^{ab}-B^{ai}B^{bj}\right|}{\sum_{a=1}^n\sum_{b=1}^n \rho_a\rho_b B^{ab}}\right)

        where:
            - :math:`\mu_i` is the chemical potential of molecule :math:`i`
            - :math:`n_j` is the molecule number of molecule :math:`j`
            - :math:`k_b` is the Boltmann constant
            - :math:`\left<V\right>` is the ensemble average box volume
            - :math:`B^{ij}` is the element of :math:`Cof(\mathbf{B})` (the cofactors of **B**) for molecules :math:`i,j`

        """
        # get cofactors x number density
        cofactors_rho = self.B_cofactors() * self.rho_ij(units="molecule/nm^3")

        # get denominator of matrix calculation
        b_lower = cofactors_rho.sum(axis=tuple(range(1,cofactors_rho.ndim))) # sum over dimensions 1:end

        # get numerator of matrix calculation
        B_prod = np.empty((self.n_sys, self.n_comp, self.n_comp, self.n_comp, self.n_comp))
        for a, b, i, j in product(range(self.n_comp), repeat=4):
            B_prod[:, a, b, i, j] = self.rho_ij(units="molecule/nm^3")[:,i,j] * (self.B_cofactors()[:,a,b]*self.B_cofactors()[:,i,j] - self.B_cofactors()[:,i,a]*self.B_cofactors()[:,j,b])
        b_upper = B_prod.sum(axis=tuple(range(3,B_prod.ndim)))

        # get chemical potential with respect to mol number in target units
        b_frac = b_upper / b_lower[:,np.newaxis,np.newaxis]
        dmu_dN_mat = self.ureg.R.to(units + "/K").magnitude * self.T()[:,np.newaxis,np.newaxis] * b_frac / (self.volume() * self._B_det)[:,np.newaxis,np.newaxis]
        return dmu_dN_mat    

    def _matrix_setup(self, matrix):
        """Setup matrices for multicomponent analysis"""
        n = self.n_comp - 1
        mat_ij = matrix[:,:n,:n]
        mat_in = matrix[:,:n,n][:,:,np.newaxis]          
        mat_jn = matrix[:,n,:n][:,np.newaxis,:]            
        mat_nn = matrix[:,n,n][:,np.newaxis,np.newaxis]
        return mat_ij - mat_in - mat_jn + mat_nn
    
    def H_ij(self, units="kJ/mol"):
        r"""
        Hessian of Gibbs mixing free energy for molecules :math:`i,j`

        Parameters
        ----------
        units: str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 3D matrix of shape ``(n_sys, n_comp-1, n_comp-1)``

        Notes
        -----
        Hessian matrix, **H**, with elements for molecules :math:`i,j` is calculated as follows:

        .. math::
            H_{ij} = M_{ij} - M_{in} - M_{jn} + M_{nn}

        .. math::
            M_{ij} = \frac{RT \Delta_{ij}^{-1}}{\rho x_i x_j}

        .. math::
            \Delta_{ij} = \frac{\delta_{ij}}{\rho x_i} + \frac{1}{\rho x_n} + G_{ij} - G_{in} - G_{jn} + G_{nn}

        where:
            - **H** is the Hessian matrix
            - **G** is the KBI matrix
            - :math:`x_i` is mol fraction of molecule :math:`i`
            - :math:`\rho` is the density of each system
            
        """
        G = self.kbi_mat()  # Cache this to avoid repeated calls

        # difference between ij interactions with each other and last component
        delta_G = self._matrix_setup(G) 

        with np.errstate(divide='ignore', invalid='ignore'):
            # get Delta matrix for Hessian calc
            Delta_ij = (
                self.kd()[np.newaxis,:] * self.V_bar()[:,np.newaxis,np.newaxis] / self.mol_fr[:,np.newaxis] 
                + (self.V_bar()/(self.mol_fr[:,self.n_comp-1]))[:,np.newaxis,np.newaxis] 
                + delta_G
            )
            Delta_ij_inv = np.linalg.inv(Delta_ij)
            R = self.ureg.R.to(units + '/K').magnitude # gas constant

            # get M matrix for hessian calculation
            M_ij = Delta_ij_inv * R * self.T()[:,np.newaxis,np.newaxis] * self.V_bar()[:,np.newaxis,np.newaxis] / (self.mol_fr[:, :, np.newaxis] * self.mol_fr[:, np.newaxis,:])

        return self._matrix_setup(M_ij)
    
    def det_H_ij(self, units="kJ/mol"):
        r"""
        Determinant, :math:`|\mathbf{H}|`, of Hessian matrix.

        Parameters
        ----------
        units: str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``
        """
        return np.linalg.det(self.H_ij(units))
    
    def S0_xx_ij(self, energy_units="kJ/mol"):
        r"""
        Structure factor as q :math:`\rightarrow` 0 for composition-composition fluctuations.

        Parameters
        ----------
        energy_units: str
            Units of energy to report values in. Default is 'kJ/mol'.
        vol_units: str
            Units of volume for scattering intensity calculations. Default is nm^3/molecule.

        Returns
        -------
        np.ndarray
            A 3D matrix of shape ``(n_sys, n_comp-1, n_comp-1)``

        Notes
        -----
        The structure factor, :math:`S_{ij}(0)`, is calculated as follows:

        .. math::
            S_{ij}(0)  = RT H_{ij}^{-1}

        where:
            - :math:`H_{ij}` is the Hessian of molecules :math:`i,j`
        """
        R = self.ureg.R.to(energy_units + '/K').magnitude # gas constant
        return R * self.T()[:,np.newaxis, np.newaxis] / self.H_ij(energy_units)
    
    def drho_elec_dx(self, units="cm^3/molecule"):
        r"""
        Electron density contrast for a mixture.

        Parameters
        ----------
        units: str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 2D array with shape ``(n_comp-1, n_comp-1)``

        Notes
        -----
        The electron density contrast, :math:`\frac{\partial \rho^e}{\partial x_i}`, is calculated according to:

        .. math::
            \frac{\partial \rho^e}{\partial x_i} = \rho \left( Z_i - Z_n \right) - \overline{Z} \rho \left( \frac{V_i - V_n}{\overline{V}} \right)

        where:
            - :math:`Z_i` is the number of electrons in molecule :math:`i`
            - :math:`V_i` is the molar volume of molecule :math:`i`
            - :math:`\overline{V}` is the molar volume of each system
        """
        # calculate electron density contrast
        return (1/self.V_bar(units))[:,np.newaxis] * (self.delta_n_elec()[np.newaxis,:] - self.n_elec_bar()[:,np.newaxis] * self.delta_V(units)[np.newaxis,:] / self.V_bar(units)[:,np.newaxis])
    
    def I0(self, units="1/cm"):
        r"""
        Small angle x-ray scattering (SAXS) intensity as q :math:`\rightarrow` 0.

        Parameters
        ----------
        units: str
            Units of inverse length to report values in. Default is '1/cm'.

        Returns
        -------
        np.ndarray
            A 1D array with shape ``(n_sys)``

        Notes
        -----
        SAXS intensity, :math:`I_0`, is calculated via:

        .. math::
            I_0 = \frac{r_e^2}{\rho} \sum_{i=1}^{n-1} \sum_{j=1}^{n-1} \left(\frac{\partial \rho^e}{\partial x_i}\right) \left(\frac{\partial \rho^e}{\partial x_j}\right) S_{ij}(0)
        
        where:
            - :math:`r_e` is the electron radius
            - :math:`\rho` is density of system
            - :math:`\frac{\partial \rho^e}{\partial x_i}` is electron density contrast for molecule :math:`i`
            - :math:`S_{ij}(0)` is structure factor for molecules :math:`i,j`

        See also
        --------
        :meth:`S0_xx_ij`: Structure factor calculation
        :meth:`drho_elec_dx`: Electron density constrast calculation
        """
        # get the electron radius in desired units
        re_units = units.split('/')[1] if '/' in units else "cm"
        re = self.Q_(2.81794092E-13, units="cm").to(re_units).magnitude  # electron radius
        vol_units = f"{units.split('/')[1]}^3/molecule"
        # calculate squared of electron density constrast combinations
        drho_dx2 = self.drho_elec_dx(units=vol_units)[:, :, np.newaxis] * self.drho_elec_dx(units=vol_units)[:, np.newaxis, :]
        # calculate saxs intensity
        i0_mat = re**2 * self.V_bar(vol_units)[:, np.newaxis, np.newaxis] * drho_dx2 * self.S0_xx_ij()
        return np.nansum(i0_mat, axis=tuple(range(1, i0_mat.ndim))) # sum of 1:last_dim
    
    def dmu_dxs(self, units="kJ/mol"):
        r"""
        The derivative of the chemical potential of molecule :math:`i` with respect to mol fraction of molecule :math:`j`.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 3D matrix of shape ``(n_sys, n_comp, n_comp)``

        Notes
        -----
        Derivative of chemical potential with respect to mol fraction (:math:`\frac{\partial \mu_i}{\partial x_j}`) is calculated as follows:

        .. math::
           \frac{\partial \mu_i}{\partial x_j} = n_T \left( \frac{\partial \mu_i}{\partial n_j} - \frac{\partial \mu_i}{\partial n_n} \right)

        where:
            - :math:`\mu_i` is the chemical potential of molecule :math:`i`
            - :math:`n_j` is the molecule number of molecule :math:`j`
            - :math:`x_j` is the mol fraction of molecule :math:`j`
            - :math:`n_T` is the total number of molecules in system        
        """
        # convert to mol fraction
        dmu = self.dmu_dN(units)  # Cache this to avoid repeated calls
        n = self.n_comp-1
        
        # chemical potential deriv / mol frac for all molecules until n-1
        dmu_dxs = self.total_molecules[:,np.newaxis,np.newaxis] * (dmu[:,:n,:n] - dmu[:,:n,-1][:,:,np.newaxis])
        
        # now get the derivative for each component
        dmui_dxi = np.full_like(self.mol_fr, fill_value=np.nan)
        dmui_dxi[:,:-1] = np.diagonal(dmu_dxs, axis1=1, axis2=2)

        # calculate chemical potential deriv for last component
        sum_xi_dmui = (self.mol_fr[:,:-1] * dmui_dxi[:,:-1]).sum(axis=1)
        dmui_dxi[:,-1] = sum_xi_dmui / self.mol_fr[:,-1]
        return dmui_dxi 

    def dlngammas_dxs(self):
        r"""
        Derivative of natural logarithm of the activity coefficient of molecule :math:`i` with respect to its mol fraction.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp, n_comp)``

        Notes
        -----
        Activity coefficient derivatives, :math:`\frac{\partial \gamma_i}{\partial x_i}` are calculated as follows:

        .. math::
            \frac{\partial \ln{\gamma_i}}{\partial x_i} = \frac{1}{k_b T}\left(\frac{\partial \mu_i}{\partial x_i}\right) - \frac{1}{x_i}

        where:
            - :math:`\mu_i` is the chemical potential of molecule :math:`i`
            - :math:`\gamma_i` is the activity coefficient of molecule :math:`i`
            - :math:`x_i` is the mol fraction of molecule :math:`i`
            - :math:`k_b` is the Boltzmann constant
        """
        if '_dlngammas_dxs' not in self.__dict__:
            # convert zeros to nan to avoid, ZeroDivisionError
            nan_z = self.mol_fr.copy()
            nan_z[nan_z == 0] = np.nan

            # calculate activity derivs
            R = self.ureg.R.to("kJ/mol/K").magnitude
            self._dlngammas_dxs = (1/(R * self.T()))[:,np.newaxis] * self.dmu_dxs("kJ/mol") - 1/nan_z

        return self._dlngammas_dxs

    def _get_ref_state_dict(self, mol):
        """get reference state parameters for each molecule"""

        # get max mol fr at each composition
        z0 = self.mol_fr.copy()
        z0[np.isnan(z0)] = 0
        comp_max = z0.max(axis=1)
        # get mol index
        i = self._mol_idx(mol=mol)
        # get mask for max mol frac at each composition
        is_max = z0[:,i] == comp_max

        # create dict for ref. state values
        # if mol is max at any composition; it cannot be a 'solute'
        if np.any(is_max):
            ref_state_dict = {
                'ref_state': 'pure_component',
                'x_initial': 1.,
                'sorted_idx_val': -1,
                'weight_fn': partial(self._weight_fn, exp_mult=1)
            }
        # if solute, use inf. dil. ref state
        else:
            ref_state_dict = {
                'ref_state': 'inf_dilution',
                'x_initial': 0.,
                'sorted_idx_val': 1,
                'weight_fn': partial(self._weight_fn, exp_mult=-1)
            }
        return ref_state_dict

    def _weight_fn(self, x, exp_mult):
        try:
            return 100 ** (exp_mult * np.log10(x))
        except ValueError as ve:
            raise ValueError(f'Cannot take log of negative value. Details: {ve}.')
    
    def ref_state(self, mol):
        r"""
        Get reference state for a molecule.

        Parameters
        ----------
        mol: str
            Molecule name in ``KBThermo.unique_molecules`` names list.

        Returns
        -------
        str
            Either '`pure_component`' or '`inf_dilution`'. Molecule is considered as '`pure_component`' if for any system it is the major component in the system. 
        """
        return self._get_ref_state_dict(mol)['ref_state']
    
    def _x_initial(self, mol):
        # get boundary condition for reference state
        return self._get_ref_state_dict(mol)['x_initial']
    
    def _sort_idx_val(self, mol):
        # get the value to sort the index by for reference state
        return self._get_ref_state_dict(mol)['sorted_idx_val']
    
    def _weights(self, mol, x):
        # get weights for mol at x for reference state
        return self._get_ref_state_dict(mol)['weight_fn'](x)
    
    def integrate_dlngammas(self, integration_type='numerical', polynomial_degree=5):
        r"""
        Integrate the derivative of activity coefficients.

        Parameters
        ----------
        integration_type: str
            This determines how the integration will be performed. Options include: numerical, polynomial.
        polynomial_degree: int
            For the '`polynomial`' integration, this specifies the degree of polynomial to fit the derivatives to.

        Returns
        -------
        np.ndarray
            A 2D array with shape ``(n_sys, n_comp)``

        Notes
        -----
        Numerical integration of activity coefficient derivatives occurs through:

        .. math::
            \ln{\gamma_i}(x_i) = \int_{a_0}^{x_i} \left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right) dx_i \approx \sum_{a=a_0}^{x_i} \frac{\Delta x}{2} \left[\left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a} + \left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a \pm \Delta x}\right]

        where:
            - :math:`\gamma_i` is the activity coefficient of molecule :math:`i`
            - :math:`x_i` is the mol fraction of molecule :math:`i`
            - :math:`\Delta x` is the step size in :math:`x` between points
    
        .. note::
            The integral is approximated by a summation using the trapezoidal rule, where the upper limit of summation is :math:`x_i` and the initial condition (or reference state) is :math:`a_0`. Note that the term :math:`a \pm \Delta x` behaves differently based on the value of :math:`a_0`: if :math:`a_0 = 1` (pure component reference state), it becomes :math:`a - \Delta x`, and if :math:`a_0 = 0` (infinite dilution reference state), it becomes :math:`a + \Delta x`.

            
        Analytical integration of activity coefficient derivatives thorough polynomial fitting occurs by fitting an n-order polynomial function to :math:`\frac{\partial \ln{\gamma_i}}{\partial x_i}`.

        .. note::
            This method takes a set of mole fractions (`xi`) and the corresponding derivatives of :math:`\ln{\gamma}`, fits a polynomial of a specified degree to the derivative data, integrates the polynomial to reconstruct :math:`\ln{\gamma}`, and evaluates :math:`\ln{\gamma}` at the given mol fractions. The integration constant is chosen such that :math:`\ln{\gamma}` obeys boundary conditions of reference state.
        
        """
        integration_type = integration_type.lower()

        ln_gammas = np.full_like(self.mol_fr, fill_value=np.nan)
        for i, mol in enumerate(self.unique_molecules):
            # get x & dlng for molecule
            xi0 = self.mol_fr[:,i]
            dlng0 = self.dlngammas_dxs()[:,i]
            lng_i = np.full(len(xi0), fill_value=np.nan)

            # filter nan
            nan_mask = (~np.isnan(xi0)) & (~np.isnan(dlng0))
            xi, dlng = xi0[nan_mask], dlng0[nan_mask]

            # if len of True values == 0; no valid mols dln gamma/dxs is found.
            if sum(nan_mask) == 0:
                raise ValueError(f'No real values found for molecule {mol} in dlngammas_dxs.')
            
            # search for x-initial
            x_initial_found = np.any(np.isclose(xi, self._x_initial(mol)))
            if not x_initial_found:
                xi = np.append(xi, self.x_initial(mol))
                dlng = np.append(dlng, 0)
            
            # sort by mol fr.
            sorted_idxs = np.argsort(xi)[::self._sort_idx_val(mol)]
            xi, dlng = xi[sorted_idxs], dlng[sorted_idxs]

            # integrate
            if integration_type == 'polynomial':
                lng = self._polynomial_integration(xi, dlng, mol, polynomial_degree)
            elif integration_type == 'numerical':
                lng = self._numerical_integration(xi, dlng, mol)
            else:
                raise ValueError(f'Integration type not recognized. Must be `polynomial` or `numerical`, {integration_type} was provided.')
            
            # now prepare data for saving
            inverse_permutation = np.argsort(sorted_idxs)
            lng = lng[inverse_permutation]

            # remove ref. state if added
            if not x_initial_found:
                x_initial_idx = np.where(lng == 0)[0][0]
                lng = np.delete(lng, x_initial_idx)
            
            try:
                lng_i[nan_mask] = lng # this makes sure that shape of lng is same as xi 
                ln_gammas[:,i] = lng_i
            except ValueError as ve:
                if len(lng) != ln_gammas.shape[0]:
                    raise ValueError(f'Length mismatch between lngammas: {len(lng)} and lngammas matrix: {ln_gammas.shape[0]}. Details: {ve}.')

        return ln_gammas

    def _polynomial_integration(self, xi, dlng, mol, polynomial_degree=5):
        # use polynomial to integrate dlng_dxs.
        try:
            dlng_fit = np.poly1d(np.polyfit(xi, dlng, polynomial_degree, w=self._weights(mol, xi)))
        except ValueError as ve:
            if polynomial_degree > len(xi):
                raise ValueError(f'Not enough data points for polynomial fit. Required degree < number points. Details: {ve}.')
            elif len(xi) != len(dlng):
                raise ValueError(f'Length mismatch! Shapes of xi {(len(xi))} and dlng {(len(xi))} do not match. Details: {ve}.')
        
        # integrate polynomial function to get ln gammas
        lng_fn = dlng_fit.integ(k=0)
        yint = 0 - lng_fn(1) # adjust for lng=0 at x=1.
        lng_fn = dlng_fit.integ(k=yint)

        # check if _lngamma_fn has been initialized
        if '_lngamma_fn_dict' not in self.__dict__:
            self._lngamma_fn_dict = {}
        if '_dlngamma_fn_dict' not in self.__dict__:
            self._dlngamma_fn_dict = {}

        # add func. to dict
        self._lngamma_fn_dict[mol] = lng_fn
        self._dlngamma_fn_dict[mol] = dlng_fit

        # evalutate lng at xi
        lng = lng_fn(xi)
        return lng

    def _numerical_integration(self, xi, dlng, mol):
        # using numerical integration via trapezoid method
        try:  
            return cumulative_trapezoid(dlng, xi, initial=0)
        except Exception as e:
            raise Exception(f'Could not perform numerical integration for {mol}. Details: {e}.')
            
    def lngamma_fn(self, mol):
        r"""
        Get the integrated polynomial function used to calculate activity coefficients (if integration type is polynomial).

        Parameters
        ----------
        mol: str
            Molecule ID for a molecule in ``KBThermo.unique_molecules``

        Returns
        -------
        np.poly1d
            Polynomial function representing :math:`\ln{\gamma}` of mol
        """
        # retrieve function for ln gamma of mol
        if '_lngamma_fn_dict' not in self.__dict__:
            self.integrate_dlngammas(integration_type="polynomial")
        return self._lngamma_fn_dict[mol]
    
    def dlngamma_fn(self, mol):
        r"""
        Get the polynomial function used to fit activity coefficient derivatives (if integration type is polynomial).

        Parameters
        ----------
        mol: str
            Molecule ID for a molecule in ``KBThermo.unique_molecules``

        Returns
        -------
        np.poly1d
            Polynomial function representing :math:`\frac{\partial \ln{\gamma}}{\partial x}` of mol
        """
        # retrieve function for dln gamma of mol
        if '_dlngamma_fn_dict' not in self.__dict__:
            self.integrate_dlngammas(integration_type="polynomial")
        return self._dlngamma_fn_dict[mol]

    def lngammas(self):
        r"""
        Results of integrated activity coefficient derivatives according to instance attribute ``gamma_integration_type``.

        Returns
        -------
        np.ndarray
            Activity coefficients as a function of system compositions points according to specificied integration type.

        See also
        --------
        :meth:`integrate_dlngammas` : Integration of activity coefficient derivatives.

        """
        if '_lngammas' not in self.__dict__:
            self._lngammas = self.integrate_dlngammas(integration_type=self.gamma_integration_type, polynomial_degree=self.gamma_polynomial_degree)
        return self._lngammas

    def GE(self, units="kJ/mol"):
        r"""
        Gibbs excess free energy calculated from activity coefficients.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``

        Notes
        -----
        Excess free energy, :math:`G^E`, is calculated according to:

        .. math::
            \frac{G^E}{RT} = \sum_{i=1}^n x_i \ln{\gamma_i}
        
        where:
            - :math:`x_i` is mol fraction of molecule :math:`i`
            - :math:`\gamma_i` is activity coefficient of molecule :math:`i`        
        """
        R = self.ureg.R.to(units + "/K")
        _GE = R * self.T(units="K") * (self.mol_fr * self.lngammas()).sum(axis=1)
        return _GE.magnitude

    def GID(self, units="kJ/mol"):
        r"""
        Ideal free energy calculated from mol fractions.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``

        Notes
        -----
        Ideal free energy, :math:`G^{id}`, is calculated according to:

        .. math::
            \frac{G^{id}}{RT} = \sum_{i=1}^n x_i \ln{x_i}
        
        where:
            - :math:`x_i` is mol fraction of molecule :math:`i`
        """
        R = self.ureg.R.to(units + "/K").magnitude
        with np.errstate(divide='ignore', invalid='ignore'):
            _GID = R * self.T(units="K") * (self.mol_fr * np.log(self.mol_fr)).sum(axis=1)
        return _GID

    def GM(self, units="kJ/mol"):
        r"""
        Gibbs mixing free energy calculated from excess and ideal contributions.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``

        Notes
        -----
        Gibbs mixing free energy, :math:`\Delta G_{mix}`, is calculated according to:

        .. math::
            \Delta G_{mix} = G^E + G^{id}
        """
        return self.GE(units) + self.GID(units)

    def Hmix(self, units="kJ/mol"):
        r"""
        Mixing enthalpy (excess enthalpy) for each system in specified units.

        Parameters
        ----------
        units : str, optional
            Units for enthalpy. Default is 'kJ/mol'.    
        
        Returns
        -------
        np.ndarray
            A 1D array of mixing enthalpies for each system in specified units.

        Notes
        -----
        This is calculated as the difference between the total enthalpy and the ideal mixing enthalpy.

        .. math::
            \Delta H_{mix} = H_{total} - \sum_{i=1}^n x_i H_i^{pure}

        where:
            - :math:`H_{total}` is the total enthalpy of the system
            - :math:`x_i` is the mol fraction of molecule :math:`i`
            - :math:`H_i^{pure}` is the pure component enthalpy of molecule :math:`i`

        .. note::
            The ideal mixing enthalpy is calculated as a linear combination of pure component enthalpies
            weighted by their mol fractions, thus requiring the pure component enthalpies to be defined under the same conditions as the systems.
        """
        return np.fromiter(self._system_mixing_enthalpy(units=units).values(), dtype=float)

    def SE(self, units="kJ/mol"):
        r"""
        Excess entropy determined from Gibbs relation between enthlapy and free energy.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``

        Notes
        -----
        Excess entropy, :math:`S^{E}`, is calculated according to:

        .. math::
            S^E = \frac{\Delta H_{mix} - G^E}{T}
        """
        return (self.Hmix(units) - self.GE(units))/self.T(units="K")

    def _property_map(self, energy_units="kJ/mol"):
        # returns a dictionary of key properties from analysis
        return {
            'mol_fr': self.mol_fr,
            'kbi': self.kbi_mat(),
            'kT': self.isothermal_compressability(units=energy_units),
            'det_H': self.det_H_ij(units=energy_units),
            'dmu': self.dmu_dxs(units=energy_units),
            'dlngamma': self.dlngammas_dxs(),
            'lngamma': self.lngammas(),
            'GE': self.GE(units=energy_units),
            'GID': self.GID(units=energy_units),
            'GM': self.GM(units=energy_units),
            'SE': self.SE(units=energy_units),
            'HE': self.Hmix(units=energy_units),
            'I0': self.I0(units="1/cm"),
        }
