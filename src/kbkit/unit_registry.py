# unit_registry.py

unit_definitions = """
# Length units (standard SI units are already known to Pint)
nanometer = 1e-9 * meter = nm
angstrom = 1e-10 * meter = angstrom = Å

# Volume units
nm3 = nanometer ** 3 = nm^3 = nanometer3
angstrom3 = angstrom ** 3 = angstrom3 = Å3

# Mass units (standard SI units already known)
dalton = 1.66053906660e-27 * kilogram = Da = amu
kilodalton = 1000 * dalton = kDa

# Energy units
joule = [energy] = J
kilojoule = 1000 * joule = kJ
calorie = 4.184 * joule = cal
kilocalorie = 1000 * calorie = kcal
electronvolt = 1.602176634e-19 * joule = eV

# Pressure units
pascal = newton / meter ** 2 = Pa
bar = 1e5 * pascal = bar
atmosphere = 101325 * pascal = atm
psi = 6894.76 * pascal = psi

# Temperature (standard Kelvin and Celsius are built-in)
degree_Celsius = kelvin; offset: 273.15 = degC = celsius

# Time units (standard SI units already known)
picosecond = 1e-12 * second = ps
nanosecond = 1e-9 * second = ns

# Common MD properties dimensions (aliases)
volume = nanometer ** 3 = volume
energy = kilojoule = energy
temperature = kelvin = temperature
enthalpy = kilojoule / mole = enthalpy
heat_capacity = kilojoule / mole / kelvin = heat_capacity

# Mole and related units
mole = [substance] = mol
millimole = 1e-3 * mole = mmol

# Force units
newton = kilogram * meter / second ** 2 = N

# constants

"""

def load_unit_registry():
    """
    Load a Pint UnitRegistry with custom unit definitions and constants.

    Returns
    -------
    ureg : pint.UnitRegistry
        A Pint UnitRegistry with custom units and constants defined.
    """
    import pint
    from scipy.constants import R as R_J
    from scipy.constants import N_A

    # create a pint unit registry
    ureg = pint.UnitRegistry()
    # load definitions in docstring above
    ureg.load_definitions(unit_definitions.splitlines())

    # define known constants manually
    ureg.R = ureg.Quantity(R_J, "joule / mole / kelvin") 
    ureg.N_A = ureg.Quantity(N_A, "molecule / mole") 
    ureg.kb = ureg.Quantity(R_J/N_A, "joule / molecule / kelvin")

    return ureg

