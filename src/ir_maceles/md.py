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
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase.io import read, write

def print_energy(a):
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.4feV  Ekin = %.4feV (T=%3.0fK)''Etot = %.4feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

def write_frame(dyn, traj_file):
    dyn.atoms.write(traj_file, append=True)

class MD:
    traj_file = 'ir_traj.traj'
    logfile= 'ir_md.log'
    temperature = 300
    equilibration_steps = 5000
    timestep = 0.25
    device = 'cpu'
    def __init__(self, model_path, input_file, n_atoms, n_steps):
        self.model_path = model_path
        self.input_file = input_file
        self.n_atoms = n_atoms
        self.n_steps = n_steps

    def run_md(self):
        init_conf = read(self.input_file)
        calculator = MACECalculator_BEC(self.model_path, self.device)
        init_conf.calc = calculator

        print("Cell:", init_conf.cell)

        # Set initial velocities
        MaxwellBoltzmannDistribution(init_conf, temperature_K = self.temperature)

        # Define NVT ensemble
        NVT_damping = 10 * self.timestep * units.fs

        dyn = NPT(
            init_conf,
            timestep=self.timestep * units.fs,
            temperature_K=self.temperature,
            ttime=NVT_damping,
            pfactor=None,
            externalstress=0.0,
        )

        # Equilibration run
        dyn.run(self.equilibration_steps)


        # Attach logger
        
        dyn.attach(write_frame(dyn, traj_file), interval=1)

        dyn.attach(
            MDLogger(dyn, init_conf, self.logfile, header=True, stress=True, peratom=False, mode="w"),
            interval=100,
        )

        print("Starting simulation (NVT) ...")
        total_dP_list = []

        for step in range(self.n_steps):
            dyn.run(1)
            BEC = init_conf.calc.results.get("BEC").to(self.device).detach()
            velocity = torch.tensor(init_conf.get_velocities(), dtype=torch.float32, device=self.device)
            dP = torch.bmm(BEC, velocity.unsqueeze(-1)).squeeze(-1)
            dP_atoms = dP[0:self.n_atoms]
            total_dP = torch.sum(dP_atoms, dim=0)
            total_dP_list.append(total_dP.detach().cpu())

            del BEC, velocity, dP, total_dP
            torch.cuda.empty_cache()
            gc.collect()

        print("Complete.")
        total_dP_stack = torch.stack(total_dP_list).numpy()

        output_dict = {"total_dp": total_dP_stack}

        output_path = os.path.splitext(self.traj_file)[0] + "_polarizations.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(output_dict, f)

        print(f"Results saved to {output_path}")
        return output_path





