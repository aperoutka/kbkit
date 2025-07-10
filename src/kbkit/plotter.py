import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

from picmol.build.lib.picmol.cli_tools.mol_click import mol
plt.style.use(Path(__file__).parent / "plt_format.mplstyle")
import itertools
import difflib

from .kb import KBThermo
from .utils import *

class Plotter: ### fix all geometric  mean references!!!
    """ for plotting results from KB analysis """

    def __init__(
            self, 
            kb_obj: KBThermo, 
            x_mol: str = None,
            molecule_map: dict = None
    ):
        self.kb = kb_obj
        self.x_mol = x_mol if x_mol is not None else self.kb.unique_molecules[0]
        self.setup_folders()
        self.molecule_map = molecule_map

        super().__setattr__('_property_alias_map', {
            "lngamma": {"lngamma", "lngammas", "ln_gamma", "ln_gammas", "lng", "gammas"},
            "dlngamma": {"dlngamma", "dlngammas", "dln_gamma", "dln_gammas", "dln_gamma_dxs", "dln_gammas_dxs", "dlng_dx"},
            "lngamma_fits": {"lngamma_fits", "lngammas_fits", "lngamma_fns", "lng_fits", "gamma_fits", "fitted_gammas", "gamma_poly", "lng_polyfits"},
            "dlngamma_fits": {"dlngamma_fits", "dlngammas_fits", "dlngamma_fns", "dlng_fits", "dgamma_fits", "fitted_dgammas", "dgamma_poly", "dlng_polyfits", "dlng_dx_fits", "dlngamma_dxs_fits"},
            "mixing": {"mixing", "mix", "mix_compare", "thermo_mixing"},
            "excess": {"excess", "ex", "ex_compare", "thermo_excess"},
            "ge": {"gibbs_excess", "ge", "excess_energy"},
            "gm": {"gibbs_mixing", "gm", "mixing_energy"},
            "hmix": {"mixing_enthalpy", "enthalpy", "h", "hmix", "he"},
            "se": {"excess_entropy", "se", "entropy", "s", "s_ex", "sex"},
            "kbi": {"kbi", "kbis", "kbintegrals", "kirkwood-buff"},
            "i0": {"i0", "saxs_i0", "saxs_intensity", "saxs_i0_conc", "saxs_i0_density"},
            'det_h': {"det_h", "hessian", "det_hessian", "h_ij", "det_h_ij", "d2gm"}
        })

    def setup_folders(self):
        self.kb_dir = mkdir(os.path.join(self.kb.base_path, 'kb_analysis'))
        self.sys_dir = mkdir(os.path.join(self.kb_dir, 'system_figures'))

    def _resolve_property_key(self, value, cutoff=0.6):
        value = value.lower()    
        best_match = None
        best_score = 0
        match_to_key = {}
        # Flatten all aliases to map them back to their canonical key
        for canonical_key, aliases in self._property_alias_map.items():
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
            raise KeyError(f"No close match found for: '{value}'. Options include: {self._property_alias_map}")
        
    @property
    def molecule_map(self):
        return self._molecule_map

    @molecule_map.setter
    def molecule_map(self, map):
        if not map:
            map = {mol: mol for mol in self.kb.unique_molecules}
        self._molecule_map = map

    @property
    def unique_names(self):
        return [self.molecule_map[mol] for mol in self.kb.unique_molecules]

        
    def rdf_combos(self, n):
        return itertools.combinations_with_replacement(range(n), 2)
    
    def n_rdf_combos(self, n):
        return len(list(self.rdf_combos(n)))
    
    @property
    def x_idx(self):
        return self.kb._mol_idx(self.x_mol)
    
    def system_combinations(self):
        # get pairwise combinations of molecules for each system.
        combos = {}
        n_combos = {}
        for system in self.kb.systems:
            sys_obj = self.kb.system_properties[system]
            n_comp = len(sys_obj.topology.molecules)
            n_combos[system] = self.n_rdf_combos(n_comp)
            combos[system] = self.rdf_combos(n_comp)
        return n_combos, combos

    def get_rdf_colors(self, cmap='jet'):
        if '_color_dict' not in self.__dict__:
            from itertools import combinations_with_replacement

            # Collect all unique unordered molecule pairs across systems
            all_pairs = set()
            for system in self.kb.system_properties:
                mols = self.kb.system_properties[system].topology.molecules
                mol_ids = [m for m in mols]
                pairs = combinations_with_replacement(mol_ids, 2)
                all_pairs.update(tuple(sorted(p)) for p in pairs)

            # Assign unique colors to each pair
            all_pairs = sorted(all_pairs)
            n_pairs = len(all_pairs)
            colormap = plt.cm.get_cmap(cmap, n_pairs)
            color_map = {pair: colormap(i) for i, pair in enumerate(all_pairs)}

            # Build nested dict color_dict[mol_i][mol_j]
            color_dict = {mol: {} for mol in self.kb.unique_molecules}
            for mol_i, mol_j in all_pairs:
                color = color_map[(mol_i, mol_j)]
                color_dict[mol_i][mol_j] = color
                color_dict[mol_j][mol_i] = color  # Ensure symmetry

            self._color_dict = color_dict

        return self._color_dict

    # now for plotting functions.
    def plot_system_kbi_analysis(self, system, units=None, alpha=0.6, cmap="jet", show=False):
        # add legend to above figure.
        color_dict = self.get_rdf_colors(cmap=cmap)
        kbi_system_dict = self.kb.kbi_dict()[system]
        units = "cm^3/mol" if units is None else units

        fig, ax = plt.subplots(1, 3, figsize=(12,4))
        for mols, mol_dict in kbi_system_dict.items():
            mol_i, mol_j = mols.split('-')
            color = color_dict.get(mol_i, {}).get(mol_j)

            rkbi = self.kb.Q_(mol_dict['rkbi'], 'nm^3/molecule').to(units).magnitude
            lkbi = self.kb.Q_(mol_dict['lambda_kbi'], 'nm^3/molecule').to(units).magnitude
            lkbi_fit = self.kb.Q_(mol_dict['lambda_kbi_fit'], 'nm^3/molecule').to(units).magnitude
            kbi_inf = self.kb.Q_(mol_dict['kbi_inf'], 'nm^3/molecule').to(units).magnitude

            ax[0].plot(mol_dict['r'], mol_dict['g'], lw=3, c=color, alpha=alpha, label=mols)
            ax[1].plot(mol_dict['r'], rkbi, lw=3, c=color, alpha=alpha, label=f'G$_{{ij}}^R$: {rkbi[-1]:.4g}')
            ax[2].plot(mol_dict['lambda'], lkbi, lw=3, c=color, alpha=alpha, label=f'G$_{{ij}}^\infty$: {kbi_inf:.4g}')
            ax[2].plot(mol_dict['lambda_fit'], lkbi_fit, ls='--', lw=4, c='k')

        ax[0].set_xlabel('r / nm')
        ax[1].set_xlabel('r / nm')
        ax[2].set_xlabel('$\lambda$')
        ax[0].set_ylabel('g(r)')
        ax[1].set_ylabel(f'G$_{{ij}}^R$ / {format_unit_str(units)}')
        ax[2].set_ylabel(f'$\lambda$ G$_{{ij}}^R$ / {format_unit_str(units)}')
        ax[0].legend(loc='lower center', bbox_to_anchor=(0.5,1.01), ncol=1, fontsize='small', fancybox=True, shadow=True)
        ax[1].legend(loc='lower center', bbox_to_anchor=(0.5,1.01), ncol=2, fontsize='small', fancybox=True, shadow=True)
        ax[2].legend(loc='lower center', bbox_to_anchor=(0.5,1.01), ncol=2, fontsize='small', fancybox=True, shadow=True)
        plt.savefig(os.path.join(self.sys_dir, f'{system}_rdfs_kbis.png'))
        if show:
            plt.show()
        else:
            plt.close()

    def plot_rdf_kbis(self, units="cm^3/mol", show=False):
        for system in self.kb.systems:
            self.plot_system_kbi_analysis(system, units=units, show=show)

    def plot_system_rdf(self, system, xlim=None, ylim=None, line=False, cmap="jet", alpha=0.6, show=True):
        # set up main fig/axes
        fig, main_ax = plt.subplots(figsize=(5,4))
        main_ax.set_box_aspect(0.6) 
        xlim = (4,5) if xlim is None else xlim 
        ylim = (0.99,1.01) if ylim is None else ylim
        inset_ax = main_ax.inset_axes(
            [0.65, 0.12, 0.3, 0.3],  # [x, y, width, height] w.r.t. axes
            xlim=xlim, ylim=ylim, # sets viewport &amp; tells relation to main axes
            # xticklabels=[], yticklabels=[]
        )
        inset_ax.tick_params(axis='x', labelsize=11)
        inset_ax.tick_params(axis='y', labelsize=11)

        color_dict = self.get_rdf_colors(cmap=cmap)
        kbi_system_dict = self.kb.kbi_dict()[system]

        for mols, mol_dict in kbi_system_dict.items():
            mol_i, mol_j = mols.split('-')
            color = color_dict.get(mol_i, {}).get(mol_j)

            # add plot content
            for ax in main_ax, inset_ax:
                ax.plot(mol_dict['r'], mol_dict['g'], c=color, alpha=alpha, label=mols)  # first example line

            # add zoom leaders
            main_ax.indicate_inset_zoom(inset_ax, edgecolor="black")

        if line:
            inset_ax.axhline(1., c='k', ls='--', lw=1.5)

        main_ax.set_xlabel('r / nm')
        main_ax.set_ylabel('g(r)')
        main_ax.legend(loc='lower center', bbox_to_anchor=(0.5,1.01), ncol=2, fontsize='small', fancybox=True, shadow=True)
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_kbis(self, units='cm^3/mol', cmap="jet", show=False):
        color_dict = self.get_rdf_colors(cmap=cmap)
        fig, ax = plt.subplots(1, 1, figsize=(5,4))
        legend_info = {}
        for system in self.kb.kbi_dict():
            for mols, mol_dict in self.kb.kbi_dict()[system].items():
                mol_i, mol_j = mols.split('-')
                i, j = [self.kb._mol_idx(mol) for mol in (mol_i, mol_j)]
                color = color_dict.get(mol_i, {}).get(mol_j)
                kbi = self.kb.kbi_mat()[:,i,j]
                kbi = self.kb.Q_(kbi, "nm^3/molecule").to(units)
                line = ax.scatter(self.kb.mol_fr[:,self.x_idx], kbi, c=color, marker='s', lw=1.8, label=mols)
                if mols not in legend_info:
                    legend_info[mols] = line
        lines = list(legend_info.values())
        labels = list(legend_info.keys())
        ax.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5,1.01), ncol=2, fontsize='small', fancybox=True, shadow=True)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks(ticks=np.arange(0,1.1,0.1))
        ax.set_xlabel(f'x$_{{{self.molecule_map[self.x_mol]}}}$')
        ax.set_ylabel(f'G$_{{ij}}^{{\infty}}$ / {format_unit_str(units)}')
        plt.savefig(self.kb_dir + f'/composition_kbi_{units.replace('^','').replace('/','_')}.png')
        if show:
            plt.show()
        else:
            plt.close()

    def get_plot_spec(self, prop: str, energy_units="kJ/mol"): # make sure all thermo properties are included here!!!
        if prop == "lngamma":
            return {
                "x_data": self.kb.mol_fr,
                "y_data": self.kb.lngammas(),
                "ylabel": r"$\ln \gamma_{i}$",
                "filename": "activity_coef.png",
                "fit_fns": None,
            }

        elif prop == "dlngamma":
            return {
                "x_data": self.kb.mol_fr,
                "y_data": self.kb.dlngammas_dxs(),
                "ylabel": r"$\partial \ln(\gamma_{i})$ / $\partial x_{i}$",
                "filename": "activity_coef_derivatives.png",
                "fit_fns": None,
            }

        elif prop == "lngamma_fits":
            return {
                "x_data": self.kb.mol_fr,
                "y_data": self.kb.lngammas(),
                "ylabel": r"$\ln \gamma_{i}$",
                "filename": "activity_coef_fits.png",
                "fit_fns": {mol: self.kb.lngamma_fn(mol) for mol in self.kb.unique_molecules},
            }

        elif prop == "dlngamma_fits":
            return {
                "x_data": self.kb.mol_fr,
                "y_data": self.kb.dlngammas_dxs(),
                "ylabel": r"$\partial \ln(\gamma_{i})$ / $\partial x_{i}$",
                "filename": "activity_coef_derivatives_fits.png",
                "fit_fns": {mol: self.kb.dlngamma_fn(mol) for mol in self.kb.unique_molecules},
            }

        elif prop == "mixing" and self.kb.n_comp == 2:
            return {
                "x_data": self.kb.mol_fr[:, self.x_idx] if self.kb.n_comp == 2 else self.kb.mol_fr[:,self.x_idx],
                "y_series": [
                    (self.kb.Hmix(energy_units), "violet", "s", r"$\Delta H_{mix}$"),
                    (-self.kb.T() * self.kb.SE(energy_units), "limegreen", "o", r"$-TS^E$"),
                    (self.kb.GID(energy_units), "darkorange", "<", r"$G^{id}$"),
                    (self.kb.GM(energy_units), "mediumblue", "^", r"$\Delta G_{mix}$"),
                ],
                "ylabel": f"Contributions to $\Delta G_{{mix}}$ / {format_unit_str(energy_units)}",
                "filename": "gibbs_mixing_contributions.png",
                "multi": True,
            }

        elif prop == "excess" and self.kb.n_comp == 2:
            return {
                "x_data": self.kb.mol_fr[:, self.x_idx],
                "y_series": [
                    (self.kb.Hmix(energy_units), "violet", "s", r"$\Delta H_{mix}$"),
                    (-self.kb.T() * self.kb.SE(energy_units), "limegreen", "o", r"$-TS^E$"),
                    (self.kb.GE(energy_units), "mediumblue", "^", r"$G^E$"),
                ],
                "ylabel": f"Excess Properties / {format_unit_str(energy_units)}",
                "filename": "gibbs_excess_properties.png",
                "multi": True,
            }
        
        elif prop == "i0" and self.kb.n_comp == 2:
            return {
                "x_data": self.kb.mol_fr[:, self.x_idx],
                "y_data": self.kb.I0(units="1/cm"),
                "ylabel": f"I$_0$ / {format_unit_str('cm^{-1}')}",
                "filename": "saxs_I0.png",
            }
        
        elif prop == "det_h" and self.kb.n_comp == 2:
            return {
                "x_data": self.kb.mol_fr[:, self.x_idx],
                "y_data": self.kb.det_H_ij(units=energy_units),
                "ylabel": f"$|H_{{ij}}|$ / {format_unit_str(energy_units)}",
                "filename": "det_hessian.png",
            }

        else:
            raise ValueError(f"Unknown property: '{property}'")

    def _render_plot(self, spec, ylim=None, show=True, cmap="jet", marker="o"):

        fig, ax = plt.subplots(figsize=(5, 4))

        if spec.get("multi", False):
            x = spec["x_data"]
            for y_data, color, mk, label in spec["y_series"]:
                ax.scatter(x, y_data, c=color, marker=mk, label=label)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5,1.01), ncol=2, fontsize='small', fancybox=True, shadow=True)
        elif spec["y_data"].ndim == 1:  # single y_data for single component
            ax.scatter(spec["x_data"], spec["y_data"], c='mediumblue', marker=marker)
        else:
            x_data = spec["x_data"]
            y_data = spec["y_data"]
            fit_fns = spec.get("fit_fns", None)
            colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, self.kb.n_comp))

            for i, mol in enumerate(self.kb.unique_molecules):
                xi = x_data[:, self.x_idx] if self.kb.n_comp == 2 else x_data[:, i]
                yi = y_data[:, i]
                ax.scatter(xi, yi, c=[colors[i]], marker=marker, label=self.molecule_map[mol])

                if fit_fns:
                    fit = fit_fns[mol]
                    zplot = generate_mol_frac_matrix(n_components=self.kb.n_comp)
                    xfit = zplot[:, self.x_idx] if self.kb.n_comp == 2 else zplot[:, i]
                    ax.plot(xfit, fit(xfit), c=colors[i], lw=2)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5,1.01), ncol=2, fontsize='small', fancybox=True, shadow=True)

        ax.set_xlabel(f"x$_{{{self.molecule_map[self.x_mol]}}}$" if self.kb.n_comp == 2 else "x$_i$")
        ax.set_ylabel(spec["ylabel"])
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks(np.arange(0, 1.1, 0.1))

        if ylim:
            ax.set_ylim(*ylim)
        elif not spec.get("multi", False):
            y_max, y_min = np.nanmax(spec["y_data"]), np.nanmin(spec["y_data"])
            pad = 0.1 * (y_max - y_min) if y_max != y_min else 0.05
            y_lb = 0 if spec["y_data"].ndim == 1 else -0.05
            ax.set_ylim(min([y_lb, y_min - pad]), max([0.05, y_max + pad]))

        plt.savefig(self.kb_dir + '/' + spec["filename"])
        if show:
            plt.show()
        else:
            plt.close()

    def _render_ternary_plots(self, property_name, energy_units="kJ/mol", cmap='jet', show=False):
        _map = {
            'ge': self.kb.GE(energy_units),
            'gm': self.kb.GM(energy_units),
            'hmix': self.kb.Hmix(energy_units),
            'se': self.kb.SE(energy_units),
            'i0': self.kb.I0("1/cm"),
            'det_h': self.kb.det_H_ij(energy_units)
        }
        arr = np.asarray(_map[property_name])
        xtext, ytext, ztext = self.unique_names
        a, b, c = self.kb.mol_fr[:,0], self.kb.mol_fr[:,1], self.kb.mol_fr[:,2]

        valid_mask = (a >= 0) & (b >= 0) & (c >= 0) & ~np.isnan(arr) & ~np.isinf(arr)
        a = a[valid_mask]
        b = b[valid_mask]
        c = c[valid_mask]
        values = arr[valid_mask]

        fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': 'ternary'})
        tp = ax.tricontourf(a, b, c, values, cmap=cmap, alpha=1, aspect=25, edgecolors='none', levels=40)
        cbar = fig.colorbar(tp, ax=ax, aspect=25, label=f'{property_name} / kJ mol$^{-1}$')

        ax.set_tlabel(xtext)
        ax.set_llabel(ytext)
        ax.set_rlabel(ztext)

        # Add grid lines on top
        ax.grid(True, which='major', linestyle='-', linewidth=1, color='k')

        ax.taxis.set_major_locator(MultipleLocator(0.10))
        ax.laxis.set_major_locator(MultipleLocator(0.10))
        ax.raxis.set_major_locator(MultipleLocator(0.10))

        plt.savefig(os.path.join(self.kb_dir, f'ternary_{property_name}.png'))
        if show:
            plt.show()
        else:
            plt.close()

    def plot_thermo_property(self, thermo_property: str, units=None, show=True, **kwargs):
        """
        Master plot function. Handles property selection, data prep, and plotting.
        Automatically dispatches to ternary plot if needed.
        """        
        prop_key = self._resolve_property_key(thermo_property.lower())
        energy_units = "kJ/mol" if units is None else units
        kbi_units = "cm^3/mol" if units is None else units

        if prop_key == "kbi":
            self.plot_kbis(units=kbi_units, show=show, **kwargs)
        
        elif self.kb.n_comp == 2 or prop_key in {"lngamma", "dlngamma", "lngamma_fits", "dlngamma_fits"}:
            spec = self.get_plot_spec(prop_key, energy_units=energy_units)
            self._render_plot(spec, show=show, **kwargs)

        elif self.kb.n_comp == 3 and prop_key in {"gm", "ge", "hmix", "se", "i0", "det_h"}:
            self._render_ternary_plots(property_name=prop_key, energy_units=energy_units, show=show, **kwargs)
        
        elif self.kb.n_comp > 3:
            print(f"WARNING: plotter does not support {prop_key} for more than 3 components. ({self.kb.n_comp} components detected.)")
        
