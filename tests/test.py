import ir_maceles
from ir_maceles.initialize import initialize_solvated
from ir_maceles.md import *
from ir_maceles.infrared import pickle_plot

from ase.io import read, write

import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt


INPUT_FILE = 'water_opt.xyz'

unsolvated = read(INPUT_FILE)

for atom in unsolvated:
    print(atom)
n_atoms = len(unsolvated.positions)

solvated = initialize_solvated(INPUT_FILE, water_file = 'water_opt.xyz')

ir_md= MD('/global/home/users/chinmayr/data/SPICE_small.model', solvated, n_atoms, 200000)

ir_md.device = 'cuda'
ir_md.equilibration_steps = 5000

pickle_file = ir_md.run_md()

omega, ft_avg, inten = pickle_plot(pickle_file, dt=0.25, dlen=10000, length=8000, window_size=2, sigma=5)

plt.plot(omega, inten)
plt.savefig('testfig.png')
