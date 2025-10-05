import sys
import os
import numpy as np

from ase import units
from ase.io import read, write
from ase.data import vdw_radii
 
def initialize_solvated(INPUT_FILE, water_file='water.pdb'):
    unsolvated = read(INPUT_FILE)
    n_atoms = len(unsolvated)

    output_file = "solvated_config.pdb"

    if not os.path.exists(water_file):
        h2o = molecule("H2O")
        write(water_file, h2o)

    n_water = 100
    box_size = box_sizer(n_water, unsolvated)
    packmol_input(output_file, solute_file, water_file, n_water, box_size)
    # Run Packmol (assumes `packmol` in PATH)
    subprocess.run(["packmol"], input=open("packmol.inp","r").read(), text=True)
    return output_file
    
def solute_volume(atoms):
    radii = np.array([vdw_radii[symbol] for symbol in atoms.get_atomic_numbers()])
    volumes = (4/3) * np.pi * radii**3
    return np.sum(volumes)
    
def box_sizer(n_water, solute_atoms, density=1.0):
    V_solute = solute_volume(solute_atoms) 
    N_A = 6.022e23
    molar_mass_H2O = 18.015
    rho_mol = density / molar_mass_H2O * N_A / 1e24 
    V_box = n_water / rho_mol + V_solute
    return V_box ** (1/3)
    
def packmol_input(output_file, solute_file, water_file, n_water, box_size):
    packmol_input = """tolerance 2.0
    filetype pdb
    output {output}
    
    structure {solute}
      number 1
      fixed 0. 0. 0. 0. 0. 0.
    end structure
    
    structure {water}
      number {n_water}
      inside box 0.0 0.0 0.0 {box_size} {box_size} {box_size}
    end structure
    """.format(output=output_file, solute=solute_file, water=water_file, n_water=n_water, box_size = box_size)
    with open("packmol.inp", "w") as f:
        f.write(packmol_input)
    return

