import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import time

from ir-maceles.mace import MACECalculator_BEC

from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md import MDLogger
from ase.io import read, write

def print_energy(a):
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.4feV  Ekin = %.4feV (T=%3.0fK)  '
          'Etot = %.4feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

def write_frame(dyn, traj_file):
        dyn.atoms.write(traj_file, append=True)
        
def run_md(model_path, traj_file, logfile, init_conf, temperature=300, equilibration_steps=5000, time_step=0.25, device='cpu'):

	calculator = MACECalculator_BEC(model_path='/global/home/users/chinmayr/data/SPICE_small.model', device)

	init_conf.set_calculator(calculator)

	from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

	#Set initial velocities using Maxwell-Boltzmann distribution,
	MaxwellBoltzmannDistribution(init_conf, temperature * units.kB)
	
	#Define the NPT ensemble,
	NPTdamping_timescale = 100 * timestep * units.fs  # Time constant for NPT dynamics
	NVTdamping_timescale = 10 * timestep * units.fs # Time constant for NVT dynamics (NPT includes both)
	dyn = NPT(init_conf, timestep = timestep * units.fs, temperature_K=temperature, ttime=NVTdamping_timescale, pfactor=None,
          externalstress=0.0) #NVT setting
	dyn.run(equilibration_steps)

	dyn.attach(write_frame(dyn, traj_file), interval=1)

	dyn.attach(MDLogger(dyn, init_conf, logfile, header=True, stress=True,
           peratom=False, mode="w"), interval=100)
           
	#Run the MD simulation,
	nsteps = 200000
	print("Starting simulation (NVT) ...")
	for step in range(nsteps):
    	dyn.run(1)
    	BEC = init_conf.calc.results.get("BEC")
    	velocity = torch.tensor(init_conf.get_velocities(), dtype=torch.float32, device=DEVICE)
    	dP = torch.bmm(BEC, velocity.unsqueeze(-1)).squeeze(-1)
    	
	print("complete.")







