import numpy as np
import os
import re
from pint import Quantity
from uncertainties import ufloat
from uncertainties.umath import *
import itertools
import glob
from natsort import natsorted


def mkdir(dir_path):
    """
    Create a new directory if path does not exist.

    Parameters
    ----------
    dir_path: str
        Path to create new directory.

    Returns
    -------
    str
        Path to new directory.
    """
    try:
        # search if path exists
        if not os.path.exists(dir_path):
            # if path not found, create directory
            os.mkdir(dir_path)
        return dir_path # return path
    except Exception as e:
        raise RuntimeError(f"Failed to create directory '{dir_path}': {e}")

def _find_file(syspath, suffix, ensemble="npt"):
    """
    Finds files in `syspath` that contain `suffix` and `ensemble`

    Parameters
    ----------
    syspath: str
        Path to simulation system containing GROMACS files
    suffix: str
        Type of file to look for, i.e., '`edr`', '`gro`', '`top`'.
    ensemble: str
        Molecular dynamics ensemble. Options: '`'npt`', '`nvt`'. Default 'npt'.
    
    Returns
    -------
    list
        List of files that match pattern.
    """
    """find a file in syspath with ensemble in filename"""
    try:
        # create pattern and use glob to find files matching pattern
        pattern = os.path.join(syspath, f"*{ensemble}*{suffix}")
        files = glob.glob(pattern)
        # Exclude equilibration/init files
        filtered = [f for f in files if not any(x in f for x in ("init", "eqm"))]
        # return empty list if no files left after filter
        if not filtered:
            return []
        else:
            # natural sort files by name
            return natsorted(filtered)
    except Exception as e:
        raise RuntimeError(f"Error finding files in '{syspath}' with suffix '{suffix}' and ensemble '{ensemble}': {e}")

def _str_to_latex_math(text: str) -> str:
    """
    Convert a string representing mathematical expressions and units into LaTeX math format.

    This function performs several transformations to convert Python-style mathematical notation
    and unit expressions into LaTeX math format:


    Parameters
    ----------
    text : str
        The input string containing mathematical expressions and units.

    Returns
    -------
    str
        The input string converted to LaTeX math format.
    """
    try:
        def inverse_fix(match):
            """replace /unit ** exponent with /unit^{exponent}"""
            unit = match.group(1)
            exp = match.group(2)
            return f"/{unit}^{{{exp}}}"

        # correct inverse unit format of first type
        text = re.sub(r'/\s*([a-zA-Z]+)\s*\*\*\s*(\d+)', inverse_fix, text)

        def inverse_unit_repl(match):
            """inverse replacement for /unit^{exp} or /unitexp to unit^{-exp}"""
            unit = match.group(1)
            m_exp = re.match(r'^([a-zA-Z]+)\^\{(-?\d+)\}$', unit)
            if m_exp:
                letters, exponent = m_exp.groups()
                new_exp = str(-int(exponent))
                return rf"\text{{ }}\mathrm{{{letters}^{{{new_exp}}}}}"
            m_simple = re.match(r'^([a-zA-Z]+)(\d+)$', unit)
            if m_simple:
                letters, digits = m_simple.groups()
                return rf"\text{{ }}\mathrm{{{letters}^{{-{digits}}}}}"
            return rf"\text{{ }}\mathrm{{{unit}^{{-1}}}}"

        # replace /unit^{exp} to unit^{-exp}
        text = re.sub(r'/\s*([a-zA-Z0-9_\^\{\}]+)', inverse_unit_repl, text)

        # convert superscripts **exp to ^{exp}
        text = re.sub(r'\*\*\s*(\(?[^\s\)]+(?:[^\s]*?)\)?)', r'^{\1}', text)

        # convert subscripts to _{val}
        text = re.sub(r'_(\(?[a-zA-Z0-9+\-*/=]+\)?)', r'_{\1}', text)

        # wrap with $ if needed
        if not (text.startswith('$') and text.endswith('$')):
            text = f"${text}$"

        return text
    
    except Exception as e:
        raise RuntimeError(f"Error converting string to LaTeX math: {e}")

def format_unit_str(text):
    """Format a unit string: convert unit words to symbols.

    If given a Pint Quantity, extract its units and format them in short form.
    The result is converted to LaTeX-friendly math for plotting.

    Parameters
    ----------
    text : pint.Quantity or str
        The quantity or unit string to format.

    Returns
    -------
    str
        A LaTeX math string representing the units.
    """
    # check if text is pint.Quantity
    if isinstance(text, Quantity):
        # convert units to str
        text = format(text.units, "~")
  
    # check that object is string
    try:
        text = str(text)
    except TypeError:
        raise TypeError(f'Could not convert type {type(text)} to str.')
    
    # format text for plotting
    unit_str = _str_to_latex_math(text)
    return unit_str

def format_quantity(quantity: Quantity):
    """
    Format a Pint Quantity (with or without uncertainty) as a string.

    Parameters
    ----------
    quantity : Quantity
        A Pint Quantity object, which may contain a nominal value and uncertainty,
        or an array of values.

    Returns
    -------
    str
        A formatted string representing the quantity, including its value (with uncertainty
        if available) and units.

    Notes
    -----
    - If the magnitude of the quantity is a NumPy array, the function computes the mean and
      standard deviation, and represents the quantity as an uncertain value.
    - If the magnitude has a `std_dev` attribute, the uncertainty is included in the output.
    - The output string uses the format: "<value> ± <uncertainty> <units>" if uncertainty is present,
      otherwise "<value> <units>".
    """
    # check type of quantity
    if not isinstance(quantity, Quantity):
        raise TypeError("Expected a Pint Quantity")
    
    # get value and units
    mag = quantity.magnitude
    unit_str = format_unit_str(quantity)
    
    # get avg/std for values depending on magnitude type
    if isinstance(mag, np.ndarray):
        mag = Quantity(ufloat(mag.mean(), mag.std()), quantity.units)
    
    # use standard deviation if available
    if hasattr(mag, 'std_dev'):
        val_str = f"{mag.nominal_value:.4g} ± {mag.std_dev:.2g}"
    else:
        val_str = f"{mag:.4g}"
    return f"{val_str} {unit_str}"


def generate_mol_frac_matrix(n_components):
    """
    Generates a matrix of all possible mol fraction compositions for a mixture with a specified number of components,
    where each composition sums to 1.0 and each component's mole fraction is chosen from a discrete set of values.
    
    Parameters
    ----------
    n_components : int
        The number of components in the mixture (must be >= 2).
    
    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (n_compositions, n_components), where each row represents a valid composition
        (mol fractions) that sum to 1.0. If no valid compositions are found, returns an empty array.
    
    Notes
    -----
    - The spacing between possible mol fraction values is determined by `n_components` using a predefined map.
    - Compositions are filtered to ensure the sum of mol fractions is close to 1.0 within a calculated tolerance.
    - The output is rounded to a consistent precision and contains only unique compositions.
    """

    # map number of components with spacing in mol fractions
    spacing_map = {2: 0.01, 3: 0.05, 4: 0.1, 5: 0.2, 6: 0.2}
    # generate 1D array of x values
    possible_x_values = np.arange(0,1+1e-5,spacing_map[n_components])

    # --- Input Validation and Preprocessing ---
    if not isinstance(n_components, int) or n_components < 2:
        raise ValueError("n_components must be an integer >= 2.")
    if not isinstance(possible_x_values, np.ndarray) or possible_x_values.ndim != 1:
        raise TypeError("possible_x_values must be a 1D NumPy array.")
    if not np.all((possible_x_values >= 0) & (possible_x_values <= 1)):
        raise ValueError("All values in possible_x_values must be between 0 and 1.")

    # Ensure unique and sorted values, pre-round for precision
    possible_x_values = np.unique(np.round(possible_x_values, 10))
    if len(possible_x_values) == 0:
            return np.array([]) # No possible values means no compositions

    # Determine precision for sum comparison
    if len(possible_x_values) > 1:
        min_diff = np.min(np.diff(possible_x_values[possible_x_values > 0]))
        tolerance = min_diff / 100 if min_diff > 0 else 1e-9
    else: # Only one possible value (e.g., [0] or [1])
        tolerance = 1e-9
            
    # Determine rounding digits for final output
    if np.any(possible_x_values > 0):
        round_ndigits = np.max([-int(np.floor(np.log10(x))) for x in possible_x_values if x > 0]) + 2
    else:
        round_ndigits = 5 # Default if only 0 is possible

    # 1. Generate all combinations of n_components from possible_x_values
    #    *([possible_x_values] * n_components)` unpacks the list into n_components arguments
    all_combinations = itertools.product(*([possible_x_values] * n_components))

    # 2. Filter combinations where the sum is close to 1.0
    #    Uses a generator expression for memory efficiency before conversion to list
    valid_compositions_gen = (
        np.array(combo) # Convert tuple to NumPy array for sum
        for combo in all_combinations
        if np.isclose(sum(combo), 1.0, atol=tolerance, rtol=0) # Check sum
    )
    
    # Convert generator to list and then to NumPy array
    valid_compositions_list = list(valid_compositions_gen)
    if not valid_compositions_list:
        return np.array([]) # No valid compositions found
    result_array = np.array(valid_compositions_list)
    
    # 3. Round to consistent precision and ensure uniqueness
    result_array = np.round(result_array, round_ndigits)
    final_unique_compositions = np.unique(result_array, axis=0)
    return final_unique_compositions

