import sys
import os
import gc
import pickle
import numpy as np
import torch
import torch.nn as nn
import time

from ir_maceles.mace import MACECalculator_BEC

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
	print('Energy per atom: Epot = %.4feV  Ekin = %.4feV (T=%3.0fK)''Etot = %.4feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

def write_frame(dyn, traj_file):
		dyn.atoms.write(traj_file, append=True)

def run_md(model_path, input_file, n_atoms, n_steps, traj_file='ir_traj.traj', logfile='ir_md.log', temperature=300, equilibration_steps=5000, time_step=0.25, device='cpu'):

	init_conf = read(input_file)
	calculator = MACECalculator_BEC(model_path, device)

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
           
	#Run the MD simulation

	print("Starting simulation (NVT) ...")

	total_dP_list = []

	for step in range(nsteps):

		dyn.run(1)

		BEC = init_conf.calc.results.get("BEC").to(device).detach()

		velocity = torch.tensor(init_conf.get_velocities(), dtype=torch.float32, device=device)
		dP = torch.bmm(BEC, velocity.unsqueeze(-1)).squeeze(-1)
		dP_atoms = dP[0:(n_atoms-1)]
		total_dP = torch.sum(dP_atoms, dim=0)
		total_dP_list.append(total_dP.detach().cpu())
		del BEC, velocity, dP, total_dP
		torch.cuda.empty_cache()
		gc.collect()

	print("complete.")

	total_dP_stack = np.array(torch.stack(total_dP_list))
	print('save dict')

	output_dict = {
		'total_dp': total_dP_stack
	}

	with open(f'{traj_file[:-5]}_polarizations.pkl', 'wb') as f:
		pickle.dump(output_dict, f)
	return f'{traj_file[:-5]}_polarizations.pkl'





