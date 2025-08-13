import numpy as np
import os
import copy
from scipy.integrate import cumulative_trapezoid
from scipy import constants
from functools import partial
from itertools import product

from .rdf import RDF
from .kbi import KBI
from .system_set import SystemSet

class KBThermo(SystemSet):
    # for applying Kirkwood-Buff (KB) theory to calculate thermodynamic properties for entire systems.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _top_mol_idx(self, mol):
        return list(self.top_molecules).index(mol)

    def _mol_idx(self, mol):
        return list(self.unique_molecules).index(mol)

    def calculate_kbis(self):
        """
        Get Kirkwood-Buff integrals (KBI) for all systems and all pairs of molecules.

        Returns
        -------
        np.ndarray
            A 3D array of Kirkwood-Buff integrals with shape ``(n_sys, n_mols, n_mols)``,
            where:

            - ``n_sys`` — number of systems
            - ``n_mols`` — number of unique molecules

        Notes
        -----
        For each system, the KBI :math:`G_{ij}` for a pair of molecules :math:`i, j`
        is computed as:

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
                if not os.path.isdir(rdf_full_path):
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
        dict
            A nested dictionary where keys are system names and values are dictionaries
            containing RDF properties for each molecular pair. Each inner dictionary has keys:

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
            A 3D array representing the original KBI matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.
        
        Returns
        -------
        np.ndarray
            A 3D array representing the modified KBI matrix with additional rows and columns for salt pairs.
            
        Notes
        -----
        - If no salt pairs are defined, it returns the original KBI matrix.
        - The salt pairs are defined in ``KBThermo.salt_pairs``, which should be a list of tuples containing the names of the salt components.
        
        This method calculates the KBI matrix for systems with salts for salt-salt interactions (:math:`G_{ss}`) and salt-other interactions (:math:`G_{si}`) as follows:

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
        
        if len(self.salt_pairs) == 0:
            return kbi_matrix

        # create new kbi-matrix
        adj = len(self.salt_pairs)-len(self.top_molecules)
        kbi_el = np.full((self.n_sys, self.n_comp+adj, self.n_comp+adj), fill_value=np.nan)

        for i, (c, a) in enumerate(self.salt_pairs):
            cj = self.top_molecules.index(c)
            aj = self.top_molecules.index(a)

            xc = self.molecule_counts[:,cj]/(self.molecule_counts[:,cj]+self.molecule_counts[:,aj])
            xa = self.molecule_counts[:,aj]/(self.molecule_counts[:,cj]+self.molecule_counts[:,aj])

            # for salt-salt interactions add to kbi-matrix
            try:
                sj = self.gm_molecules.index('-'.join([c,a]))
            except:
                sj = self.gm_molecules.index('-'.join([a,c]))
            kbi_el[sj, sj] = xc**2 * kbi_matrix[cj, cj] + xa**2 * kbi_matrix[aj, aj] + xc*xa * (kbi_matrix[cj, aj] + kbi_matrix[aj, cj])

            # for salt other interactions
            for m1, mol1 in enumerate(self.nosalt_molecules):
                m1j = self.top_molecules.index(mol1)
                for m2, mol2 in enumerate(self.nosalt_molecules):
                    m2j = self.top_molecules.index(mol2)
                    kbi_el[m1, m2] = kbi_matrix[m1j, m2j]
                # now for mol-salt interactions
                kbi_el[m1, sj] = xc * kbi_matrix[m1, cj] + xa * kbi_matrix[m1, aj]
                kbi_el[sj, m1] = xc * kbi_matrix[cj, m1] + xa * kbi_matrix[aj, m1]

        return kbi_el
    
    def kbi_mat(self):
        """
        Get the KBI matrix with electrolyte corrections applied if salt pairs are defined.

        Returns
        -------
        np.ndarray
            A 3D array representing the KBI matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of components,
            including any additional salt pairs if defined.        
        """
        if '_kbi_mat' not in self.__dict__:
            kbi_matrix = self.calculate_kbis()
            self._kbi_mat = self._electrolyte_kbi_correction(kbi_matrix=kbi_matrix.copy())
        return self._kbi_mat
    
    def kd(self):
        """
        Get the Kronecker delta matrix for the number of unique molecules. 
        
        Returns
        -------
        np.ndarray
            A 2D array representing the Kronecker delta matrix with shape ``(n_comp, n_comp)``,
            where ``n_comp`` is the number of unique components.
        """
        return np.eye(self.n_comp)
    
    def B_mat(self):
        r"""
        Construct a symmetric matrix B for each system based on the number densities and KBIs.

        Returns
        -------
        np.ndarray
            A 3D array representing the B matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Notes
        -----
        The B matrix is calculated for molecules :math:`i,j`, using the formula:

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
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.linalg.inv(self.B_mat())
    
    @property
    def _B_det(self):
        """np.ndarray: Determinant of the B matrix."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.linalg.det(self.B_mat())

    def B_cofactors(self):
        r"""
        Get the cofactors of the B matrix for each system.
        
        Returns
        -------
        np.ndarray
            A 3D array representing the cofactors of the B matrix with shape ``(n_sys, n_comp, n_comp)``,
        
        Notes
        -----
        The cofactors of matrix B (:math:`Cof(B)`) are calculated as:
        
        .. math::
            Cof(B) = |B| \cdot B^{-1}
        
        where:
            - :math:`|B|` is the determinant of the B matrix.
            - :math:`B^{-1}` is the inverse of the B matrix.
        """
        if '_B_cofactors' not in self.__dict__:
            self._B_cofactors = self._B_det[:,np.newaxis,np.newaxis] * self._B_inv
        return self._B_cofactors
    
    def A_mat(self):
        if '_A_mat' not in self.__dict__:
            self._A_mat = (1/self.V_bar(units="nm^3/molecule"))[:,np.newaxis,np.newaxis] * self.mol_fr[:,:,np.newaxis] * self.mol_fr[:,np.newaxis,:] * self.kbi_mat() + self.mol_fr[:,:,np.newaxis] * self.kd()[np.newaxis,:,:]
        return self._A_mat

    def isothermal_compressability(self, units="kJ/mol"):
        R = self.ureg.R.to(units + "/K").magnitude
        kT = (1 / (R * self.T())) * (self.molar_volume()[np.newaxis,:]/self.A_mat()[:,0,:]).sum(axis=1)
        return self.Q_(kT, units=f"{units.split('/')[1]}/{units.split('/')[0]} * nm^3/molecule").to("1/kPa").magnitude

    def dmu_dN(self, units="kJ/mol"):
        cofactors_rho = self.B_cofactors() * self.rho_ij(units="molecule/nm^3")
        b_lower = cofactors_rho.sum(axis=tuple(range(1,cofactors_rho.ndim))) # sum over dimensions 1:end

        B_prod = np.empty((self.n_sys, self.n_comp, self.n_comp, self.n_comp, self.n_comp))
        for a, b, i, j in product(range(self.n_comp), repeat=4):
            B_prod[:, a, b, i, j] = self.rho_ij(units="molecule/nm^3")[:,i,j] * (self.B_cofactors()[:,a,b]*self.B_cofactors()[:,i,j] - self.B_cofactors()[:,i,a]*self.B_cofactors()[:,j,b])
        b_upper = B_prod.sum(axis=tuple(range(3,B_prod.ndim)))

        b_frac = b_upper / b_lower[:,np.newaxis,np.newaxis]
        dmu_dN_mat = self.ureg.R.to(units + "/K").magnitude * self.T()[:,np.newaxis,np.newaxis] * b_frac / (self.volume() * self._B_det)[:,np.newaxis,np.newaxis]
        return dmu_dN_mat    

    def _matrix_setup(self, matrix):
        n = self.n_comp - 1
        mat_ij = matrix[:,:n,:n]
        mat_in = matrix[:,:n,n][:,:,np.newaxis]          
        mat_jn = matrix[:,n,:n][:,np.newaxis,:]            
        mat_nn = matrix[:,n,n][:,np.newaxis,np.newaxis]
        return mat_ij - mat_in - mat_jn + mat_nn
    
    def H_ij(self, units="kJ/mol"):
        G = self.kbi_mat()  # Cache this to avoid repeated calls
        delta_G = self._matrix_setup(G)

        with np.errstate(divide='ignore', invalid='ignore'):
            Delta_ij = (
                self.kd()[np.newaxis,:] * self.V_bar()[:,np.newaxis,np.newaxis] / self.mol_fr[:,np.newaxis] 
                + (self.V_bar()/(self.mol_fr[:,self.n_comp-1]))[:,np.newaxis,np.newaxis] 
                + delta_G
            )
            Delta_ij_inv = np.linalg.inv(Delta_ij)

            R = self.ureg.R.to(units + '/K').magnitude
            M_ij = Delta_ij_inv * R * self.T()[:,np.newaxis,np.newaxis] * self.V_bar()[:,np.newaxis,np.newaxis] / (self.mol_fr[:, :, np.newaxis] * self.mol_fr[:, np.newaxis,:])
        return self._matrix_setup(M_ij)
    
    def det_H_ij(self, units="kJ/mol"):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.linalg.det(self.H_ij(units))
    
    def S0_xx_ij(self, energy_units="kJ/mol", vol_units="nm^3/molecule"):
        R = self.ureg.R.to(energy_units + '/K').magnitude
        return self.V_bar(vol_units)[:, np.newaxis, np.newaxis] * R * self.T()[:,np.newaxis, np.newaxis] / self.H_ij(energy_units)
    
    def drho_elec_dx(self, units="cm^3/molecule"):
        # calculate electron density contrast
        drho_dx = (1/self.V_bar(units))[:,np.newaxis] * (self.delta_n_elec()[np.newaxis,:] - self.n_elec_bar()[:,np.newaxis] * self.delta_V(units)[np.newaxis,:] / self.V_bar(units)[:,np.newaxis])
        return np.nansum(drho_dx, axis=1)
    
    def I0(self, units="1/cm"):
        re_units = units.split('/')[1] if '/' in units else "cm"
        re = self.Q_(2.81794092E-13, units="cm").to(re_units).magnitude  # electron radius
        vol_units = f"{units.split('/')[1]}^3/molecule"
        _I0 = re**2 * self.drho_elec_dx(units=vol_units)[:, np.newaxis, np.newaxis]**2 * self.S0_xx_ij(vol_units=vol_units)
        return np.nansum(_I0, axis=tuple(range(1, _I0.ndim)))
    
    def dmu_dxs(self, units="kJ/mol"):
        # convert to mol fraction
        dmu = self.dmu_dN(units)  # Cache this to avoid repeated calls
        n = self.n_comp-1
        dmu_dxs = self.total_molecules[:,np.newaxis,np.newaxis] * (dmu[:,:n,:n] - dmu[:,:n,-1][:,:,np.newaxis])
        # now get the derivative for each component
        dmui_dxi = np.full_like(self.mol_fr, fill_value=np.nan)
        dmui_dxi[:,:-1] = np.diagonal(dmu_dxs, axis1=1, axis2=2)
        sum_xi_dmui = (self.mol_fr[:,:-1] * dmui_dxi[:,:-1]).sum(axis=1)
        dmui_dxi[:,-1] = sum_xi_dmui / self.mol_fr[:,-1]
        return dmui_dxi 

    def dlngammas_dxs(self):
        if '_dlngammas_dxs' not in self.__dict__:
            # convert zeros to nan to avoid, ZeroDivisionError
            nan_z = copy.deepcopy(self.mol_fr)
            nan_z[nan_z == 0] = np.nan
            R = self.ureg.R.to("kJ/mol/K").magnitude
            self._dlngammas_dxs = (1/(R * self.T()))[:,np.newaxis] * self.dmu_dxs("kJ/mol") - 1/nan_z
        return self._dlngammas_dxs

    def _get_ref_state_dict(self, mol):
        # get max mol fr at each composition
        z0 = copy.deepcopy(self.mol_fr)
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
        return self._get_ref_state_dict(mol)['ref_state']
    
    def x_initial(self, mol):
        return self._get_ref_state_dict(mol)['x_initial']
    
    def sort_idx_val(self, mol):
        return self._get_ref_state_dict(mol)['sorted_idx_val']
    
    def weights(self, mol, x):
        return self._get_ref_state_dict(mol)['weight_fn'](x)
    
    def integrate_dlngammas(self, integration_type='numerical', polynomial_degree=5):
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
            x_initial_found = np.any(np.isclose(xi, self.x_initial(mol)))
            if not x_initial_found:
                xi = np.append(xi, self.x_initial(mol))
                dlng = np.append(dlng, 0)
            
            # sort by mol fr.
            sorted_idxs = np.argsort(xi)[::self.sort_idx_val(mol)]
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
            dlng_fit = np.poly1d(np.polyfit(xi, dlng, polynomial_degree, w=self.weights(mol, xi)))
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
        # retrieve function for ln gamma of mol
        if '_lngamma_fn_dict' not in self.__dict__:
            self.integrate_dlngammas(integration_type="polynomial")
        return self._lngamma_fn_dict[mol]
    
    def dlngamma_fn(self, mol):
        # retrieve function for dln gamma of mol
        if '_dlngamma_fn_dict' not in self.__dict__:
            self.integrate_dlngammas(integration_type="polynomial")
        return self._dlngamma_fn_dict[mol]

    def lngammas(self):
        if '_lngammas' not in self.__dict__:
            self._lngammas = self.integrate_dlngammas(integration_type=self.gamma_integration_type, polynomial_degree=self.gamma_polynomial_degree)
        return self._lngammas

    def GE(self, units="kJ/mol"):
        R = self.ureg.R.to(units + "/K")
        _GE = R * self.T(units="K") * (self.mol_fr * self.lngammas()).sum(axis=1)
        return _GE.magnitude

    def GID(self, units="kJ/mol"):
        R = self.ureg.R.to(units + "/K").magnitude
        with np.errstate(divide='ignore', invalid='ignore'):
            _GID = R * self.T(units="K") * (self.mol_fr * np.log(self.mol_fr)).sum(axis=1)
        return _GID

    def GM(self, units="kJ/mol"):
        return self.GE(units) + self.GID(units)

    def SE(self, units="kJ/mol"):
        return (self.Hmix(units) - self.GE(units))/self.T(units="K")

    def _property_map(self, energy_units="kJ/mol"):
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
