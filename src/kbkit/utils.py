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
    # creates a new directory and assigns it a variable
    if os.path.exists(dir_path) == False:
        os.mkdir(dir_path)
    return dir_path

def _find_file(syspath, suffix, ensemble="npt"):
    """find a file in syspath with ensemble in filename"""
    pattern = os.path.join(syspath, f"*{ensemble}*{suffix}")
    files = glob.glob(pattern)
    # Exclude equilibration/init files
    filtered = [f for f in files if not any(x in f for x in ("init", "eqm"))]
    if not filtered:
        return filtered
    else:
        return natsorted(filtered)

def _str_to_latex_math(text: str) -> str:
    """convert str to latex math"""
    # Step 1: Replace /unit ** exponent with /unit^{exponent}
    def inverse_fix(match):
        unit = match.group(1)
        exp = match.group(2)
        return f"/{unit}^{{{exp}}}"

    text = re.sub(r'/\s*([a-zA-Z]+)\s*\*\*\s*(\d+)', inverse_fix, text)

    # Step 2: Now inverse replacement on /unit^{exp} or /unitexp
    def inverse_unit_repl(match):
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

    text = re.sub(r'/\s*([a-zA-Z0-9_\^\{\}]+)', inverse_unit_repl, text)

    # Step 3: Superscripts: convert **exp to ^{exp}
    text = re.sub(r'\*\*\s*(\(?[^\s\)]+(?:[^\s]*?)\)?)', r'^{\1}', text)

    # Step 4: Subscripts
    text = re.sub(r'_(\(?[a-zA-Z0-9+\-*/=]+\)?)', r'_{\1}', text)

    # Step 5: wrap with $ if needed
    if not (text.startswith('$') and text.endswith('$')):
        text = f"${text}$"

    return text

def format_unit_str(text):
    """format unit string --> convert words to symbols"""
    if isinstance(text, Quantity):
        text = text.units.format_babel(locale="en", spec="~")
  
    # check that object is string
    try:
        text = str(text)
    except TypeError:
        raise TypeError(f'Could not convert type {type(text)} to str.')
    
    # format text for plotting
    unit_str = _str_to_latex_math(text)
    return unit_str

def format_quantity(quantity: Quantity):
    """format a pint Quantity (with or without uncertainty)"""
    # check type of quantity
    if not isinstance(quantity, Quantity):
        raise TypeError("Expected a Pint Quantity")
    
    # get value and units
    mag = quantity.magnitude
    unit_str = format_unit_str(quantity)
    
    # get avg/std for values depending on magnitude type
    if isinstance(mag, np.ndarray):
        avg = np.mean(mag)
        std = np.std(mag)
        mag = Quantity(ufloat(avg, std), quantity.units)
    
    # use standard deviation if available
    if hasattr(mag, 'std_dev'):
        val_str = f"{mag.nominal_value:.4g} ± {mag.std_dev:.2g}"
    else:
        val_str = f"{mag:.4g}"
    return f"{val_str} {unit_str}"


def generate_mol_frac_matrix(n_components):

    spacing_map = {2: 0.01, 3: 0.05, 4: 0.1, 5: 0.2, 6: 0.2}
    possible_x_values = np.arange(0,1+1e-5,spacing_map[n_components])

    # --- Input Validation and Preprocessing (Similar to previous version) ---
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError("n_components must be an integer >= 1.")
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

