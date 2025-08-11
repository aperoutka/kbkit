import os
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use(Path(__file__).parent.parent / "presentation.mplstyle")

class RDF:
    def __init__(self, rdf_file, rdf_convergence=(5e-3, 5e-3)):
        self.rdf_file = rdf_file
        # read rdf_file
        self._read()
        # make sure rdf is converged
        self.convergence_check(*rdf_convergence)

    def _read(self):
        # sets r & g properties from rdf filepath.
        try:
            r, g = np.loadtxt(self.rdf_file, comments=["@","#"], unpack=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"RDF file: '{self.rdf_file}' not found.")
        except IOError as ioe:
            raise IOError(f"Error reading file: '{self.rdf_file}': {ioe}.")

        # filter tail noise
        gstd = np.array([np.nanstd(g[i-1:i+1]) for i in range(1, len(g)+1)])
        mask = ((gstd > 0.01) | (gstd == 0)) & (r > r.max() - 1)
        self._r = r[~mask]
        self._g = g[~mask]
        self._rmin = self._r.max() - 1
        
    @property
    def r(self):
        return self._r

    @property
    def rmax(self):
        return self._r.max()
    
    @property
    def g(self):
        return self._g

    @property
    def rmin(self):
        return self._rmin
    
    @rmin.setter
    def rmin(self, value):
        if value < 0:
            raise ValueError("Lower bound must be non-negative.")
        if value > self.rmax:
            raise ValueError(f"Lower bound {value} exceeds rmax {self.rmax}.")
        self._rmin = value

    @property
    def r_mask(self):
        return(self.r >= self.rmin) & (self.r <= self.rmax)

    def convergence_check(
        self, 
        convergence_threshold=5e-3, 
        flatness_threshold=5e-3,
        max_attempts=10,
    ):
        for _ in range(max_attempts):
            r = self._r[self.r_mask]
            g = self._g[self.r_mask]

            if len(r) < 3:
                raise ValueError("Not enough points for convergence check.")

            slope, _ = np.polyfit(r, g, 1)
            std_dev = np.nanstd(g)

            if abs(slope) < convergence_threshold and std_dev < flatness_threshold:
                return True

            # Adjust rmin to expand cutoff region slightly
            self.rmin += 0.1 * (self.rmax - self.rmin)
            if self.rmin >= self.rmax - 0.1:
                break

        print(
            f"Convergence not achieved after {max_attempts} attempts for {os.path.basename(self.rdf_file)} "
            f"in system {os.path.basename(os.path.dirname(os.path.dirname(self.rdf_file)))}; "
            f"slope (thresh={convergence_threshold}) {slope:.4g}, "
            f"stdev (thresh={flatness_threshold}) {std_dev:.4g}, "
        )
        self.rmin = self.rmax - 0.1  # reset rmin to max possible safe value
        return False

    def plot(self, xlim=[4,5], ylim=[0.99,1.01], line=False, save_dir=None):
        # set up main fig/axes
        fig, main_ax = plt.subplots()
        main_ax.set_box_aspect(0.6) 
        inset_ax = main_ax.inset_axes(
            [0.65, 0.12, 0.3, 0.3],  # [x, y, width, height] w.r.t. axes
            xlim=xlim, ylim=ylim, # sets viewport &amp; tells relation to main axes
            # xticklabels=[], yticklabels=[]
        )
        inset_ax.tick_params(axis='x', labelsize=11)
        inset_ax.tick_params(axis='y', labelsize=11)

        # add plot content
        for ax in main_ax, inset_ax:
            ax.plot(self.r, self.g)  # first example line
        if line:
            inset_ax.axhline(1., c='k', ls='--', lw=1.5)

        # add zoom leaders
        main_ax.indicate_inset_zoom(inset_ax, edgecolor="black")
        main_ax.set_xlabel('r / nm')
        main_ax.set_ylabel('g(r)')
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, self.rdf_file[:-4] + '.png'))
        plt.show()

    @staticmethod
    def extract_mols(rdf_file, mol_list):
        pattern = r'(' + '|'.join(re.escape(mol) for mol in mol_list) + r')'
        return re.findall(pattern, os.path.basename(rdf_file))
