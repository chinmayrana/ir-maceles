import sys

import ir_maceles
from ir_maceles.initialize import initialize_solvated
from ir_maceles.md import run_md
from ir_maceles.infrared import pickle_plot

from ase.io import read, write

import matplotlib.pyplot as plt


INPUT_FILE = 'alanine-dipeptide-nowater-1.xyz'

unsolvated = read(INPUT_FILE)

for atom in unsolvated:
    print(atom)
n_atoms = len(unsolvated.positions)

solvated = initialize_solvated(INPUT_FILE)

pickle_file = run_md('/Users/avirana/projects/research/SPICE_small.model', solvated, n_atoms, 3, equilibration_steps = 0, device='cuda')

omega, ft_avg, inten = pickle_plot(pickle_file, dt=0.25, dlen=10000, length=8000, window_size=2, sigma=5)

plt.plot(omega, inten)
plt.savefig('testfig.png')
