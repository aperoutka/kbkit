import os
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use(Path(__file__).parent.parent / "presentation.mplstyle")

class RDF:
    """
    Class to handle RDF (Radial Distribution Function) data.
    Reads RDF data from a file, checks for convergence, and provides methods to plot the RDF and extract molecular information.

    Parameters
    ----------
    rdf_file : str
        Path to the RDF file containing radial distances and corresponding g(r) values.
    rdf_convergence : tuple of float, optional
        Tuple containing convergence thresholds for slope and standard deviation of g(r). Default is (5e-3, 5e-3).    
    """
    def __init__(self, rdf_file, rdf_convergence=(5e-3, 5e-3)):
        self.rdf_file = rdf_file
        # read rdf_file
        self._read()
        # make sure rdf is converged
        self.convergence_check(*rdf_convergence)

    def _read(self):
        """Reads the RDF file and extracts radial distances (r) and g(r) values.
        The file is expected to have two columns: r and g(r).
        It filters out noise from the tail of the RDF curve.
        """
        try:
            # sets r & g properties from rdf filepath.
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
        """np.ndarray: Radial distances in nm"""
        return self._r

    @property
    def rmax(self):
        """float: Maximum radial distance in nm"""
        return self._r.max()
    
    @property
    def g(self):
        """np.ndarray: g(r) values corresponding to the radial distances"""
        return self._g

    @property
    def rmin(self):
        """float: Lower bound for the radial distance, used in convergence checks."""
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
        """np.ndarray: Boolean mask for radial distances within the range [rmin, rmax]."""
        return(self.r >= self.rmin) & (self.r <= self.rmax)

    def convergence_check(
            self, 
            convergence_threshold=5e-3, 
            flatness_threshold=5e-3,
            max_attempts=10,
        ):
        """
        Checks if the RDF is converged based on the slope of g(r) and its standard deviation.

        Parameters
        ----------
        convergence_threshold : float, optional
            Threshold for the slope of g(r) to determine convergence. Default is 5e-3.
        flatness_threshold : float, optional
            Threshold for the standard deviation of g(r) to determine flatness. Default is 5e-3.
        max_attempts : int, optional
            Maximum number of attempts to check convergence by adjusting the lower bound (rmin). Default is 10.

        Returns
        -------
        bool
            True if the RDF is converged, False otherwise.
        """
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

    def plot(
            self, 
            xlim=[4,5], 
            ylim=[0.99,1.01], 
            line=False, 
            save_dir=None
        ):
        """
        Plots the RDF with an inset showing a zoomed-in view of the specified region.

        Parameters
        ----------
        xlim : list of float, optional
            x-axis limits for the inset plot. Default is [4, 5].
        ylim : list of float, optional
            y-axis limits for the inset plot. Default is [0.99, 1.01].
        line : bool, optional
            If True, adds a horizontal line at y=1. Default is False.
        save_dir : str, optional
            Directory to save the plot. If None, the plot is displayed but not saved. Default is None.
        """
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
        """ 
        Extracts the names of molecules from the RDF file name.
        
        Parameters
        ----------
        rdf_file : str
            Path to the RDF file.
        mol_list : list of str
            List of molecule names to search for in the file name.
        """
        pattern = r'(' + '|'.join(re.escape(mol) for mol in mol_list) + r')'
        return re.findall(pattern, os.path.basename(rdf_file))
