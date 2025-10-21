import sys
import os
import subprocess

import numpy as np

from ase import units
from ase.io import read, write
from ase.data import vdw_radii
from ase.build import molecule
 
def initialize_solvated(INPUT_FILE, water_file='water.xyz'):
    unsolvated = read(INPUT_FILE)
    n_atoms = len(unsolvated)
    
    packmol_file = "packmol.xyz"

    if not os.path.exists(water_file):
        h2o = molecule("H2O")
        write(water_file, h2o)

    n_water = 150
    box_size = box_sizer(n_water, unsolvated)
    packmol_input(packmol_file, INPUT_FILE, water_file, n_water, box_size)
    # Run Packmol (assumes `packmol` in PATH)
    with open("packmol.inp") as f:
        subprocess.run(f"packmol < packmol.inp", shell=True, check=True)

    solvated = read(packmol_file)
    
    solvated.set_cell([[box_size,0,0],[0,box_size,0],[0,0,box_size]])
    solvated.set_pbc([True, True, True])
    output_file = 'output.extxyz'
    write(output_file, solvated)  
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
    half_box = box_size / 2
    packmol_input = """tolerance 2.0
    filetype xyz
    output {output}
    
    structure {solute}
      number 1
      center
      fixed 0. 0. 0. 0. 0. 0.
    end structure
    
    structure {water}
      number {n_water}
      inside box -{half_box:.3f} -{half_box:.3f} -{half_box:.3f} {half_box:.3f} {half_box:.3f} {half_box:.3f}
    end structure
    """.format(output=output_file, solute=solute_file, water=water_file, n_water=n_water, half_box=half_box)
    with open("packmol.inp", "w") as f:
        f.write(packmol_input)
    return

