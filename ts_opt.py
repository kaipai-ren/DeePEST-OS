import os
import time
import torch
import warnings
import numpy as np

from xtb.ase.calculator import XTB
from mace.calculators import MACECalculator
from ase.utils.forcecurve import fit_images, plotfromfile
from ase.io.trajectory import TrajectoryReader
from ase.calculators.mixing import SumCalculator
from sella import Sella
from sella import IRC
from x3dase.x3d import X3D
from ase.io import read
from ase.vibrations import Vibrations
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
import matplotlib.pyplot as plt
from ase.io import read, write, animation
from ase.mep.neb import NEB, NEBOptimizer, NEBTools
from ase.mep.autoneb import AutoNEB
from ase.optimize.bfgs import BFGS
from natsort import ns, natsorted

from neuralneb.painn.painn import PaiNN
from neuralneb import utils
import configparser
from argparse import ArgumentParser
import traceback
import jax

jax.config.update('jax_platform_name', 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def visualize_mep(images, save_dir, rxn_name, job_type, interval=50):
    """
    Save the MEP (Minimum Energy Path) PNG and search pathway GIF
    """
    fit = fit_images(images)
    fit.plot()
    plt.savefig(os.path.join(save_dir, f'{rxn_name}_{job_type}.png'), dpi=300)
    animation.write_animation(os.path.join(save_dir, f'{rxn_name}_{job_type}.gif'), images=images, writer='pillow', interval=interval)  # Save the gif of dimer path
    plt.close()

def read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def delete_files_with_extension(folder_path, extension):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension) and '0' in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)

def sella_refine_irc(rxn_names, input_path, output_path, calculator, redine_f_max, irc_f_max, steps):

    title = 'name\trefine_time\trefine_steps\tirc_time\tirc_steps\tEf\tEr\tflag\n'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save all the structure optimization refine and IRC running times
    if not os.path.exists(os.path.join(output_path, 'all_log.txt')):
        with open(os.path.join(output_path, 'all_log.txt'), 'a') as f:
            f.write(title)

    for fs in rxn_names:
        rxn_name = fs.split('.')[0]
        # Work folder
        rxn_output_path = os.path.join(output_path, rxn_name)
        if not os.path.exists(rxn_output_path):
            os.makedirs(rxn_output_path)
        else:
            print(f'{rxn_name} already exists, skip.')
            continue

        # Optimize transition state structure
        ref_file_dir = os.path.join(rxn_output_path, f'{rxn_name}_ref.traj')
        ref_ts_dir = os.path.join(rxn_output_path, f'{rxn_name}_ts.xyz')
        irc_file_dir = os.path.join(rxn_output_path, f'{rxn_name}_irc.traj')

        max_retries = 4  # Set the maximum retry attempts
        for attempt in range(1, max_retries + 1):
            print(f'Refining loop {attempt} for rxn_{rxn_name}...')
            try:
                if os.path.exists(ref_file_dir):
                    os.remove(ref_file_dir)
                if os.path.exists(ref_ts_dir):
                    os.remove(ref_ts_dir)
                # Read input file
                atoms = read(os.path.join(input_path, str(fs)))
                atoms.calc = calculator
                # Optimize transition state structure
                time_start = time.time()
                dyn = Sella(atoms, internal=True, trajectory=ref_file_dir, eta=1e-5, gamma=0.4)
                dyn.run(fmax=redine_f_max, steps=steps)
                time_end = time.time()
                sella_ref_time = time_end - time_start
                print(f'{rxn_name} sella_ref_time: {round(sella_ref_time, 2)} s')
                ref_time = f'{round(sella_ref_time, 2)}'
                traj = TrajectoryReader(ref_file_dir)
                refine_steps = len(traj) - 1
                write(ref_ts_dir, traj[-1])

                # Perform frequency analysis on the optimized transition state structure
                vib_dir = os.getcwd() + '/vib'  # If the vib folder exists, delete it
                if os.path.exists(vib_dir):
                    shutil.rmtree(vib_dir)
                vib = Vibrations(atoms)
                vib.run()
                vib.summary(log=os.path.join(rxn_output_path, f'{rxn_name}_vibration.log'))
                freqs = vib.get_frequencies()
                imaginary_count = np.sum(np.imag(freqs) != 0)  # Count the number of imaginary frequencies
                print(f'rxn {rxn_name} imaginary_count: {imaginary_count}')

                # Create a log file to save running time and imaginary frequency count
                with open(os.path.join(rxn_output_path, 'run.log'), 'w') as f:
                    f.write(f'rxn_name: {rxn_name}\nref_time: {ref_time}\n'
                            f'refine_steps: {refine_steps}\nimaginary_count: {imaginary_count}\n')

                if imaginary_count > 6:  # If there are more than 6 imaginary frequencies, recalculate
                    print(f'rxn {rxn_name} has more than 3 imaginary frequencies, recalculate it.')
                    continue

            except Exception as e:
                traceback.print_exc()
                print(f'refine error on {rxn_name}')
                if attempt <= max_retries:
                    print(f'Attempt {attempt} failed on {rxn_name} REFINE. Retrying...')
                    time.sleep(3)  # Add an appropriate wait time
                    continue
                else:
                    print(f'REFINE error on {rxn_name} after {max_retries} attempts.')
                    continue

            # Calculate IRC (Intrinsic Reaction Coordinate)
            print(f'Calculating IRC for {rxn_name}...')
            try:
                if os.path.exists(irc_file_dir):
                    os.remove(irc_file_dir)
                # Read optimized transition state structure
                irc_inits = traj[-1]  # read(ref_ts_dir)
                irc_inits.calc = calculator
                opt = IRC(irc_inits, trajectory=irc_file_dir, dx=0.05, eta=1e-4, gamma=0.4)  # Can adjust eta for better convergence
                time_start = time.time()
                opt.run(fmax=irc_f_max, steps=steps, direction='forward')
                flag = len(TrajectoryReader(irc_file_dir))  # Mark the division point for forward and reverse directions
                opt.run(fmax=irc_f_max, steps=steps, direction='reverse')
                time_end = time.time()
                sella_irc_time = time_end - time_start
                print(f'{rxn_name} sella_irc_time: {round(sella_irc_time, 2)} s')
                irc_time = f'\t{round(sella_irc_time, 2)}'

                # Draw trajectory of sella IRC
                irc_traj = TrajectoryReader(irc_file_dir)
                irc_steps = len(irc_traj)

                # Process IRC forward and reverse results, save them in the irc_trajectory
                arrange_irc = []
                for i in range(flag - 1, -1, -1):
                    arrange_irc.append(irc_traj[i])
                arrange_irc.extend(irc_traj[flag:])
                print(len(arrange_irc))
                print(len(irc_traj))
                # Draw the IRC optimization path in 3D structure
                try:
                    visualize_mep(arrange_irc, rxn_output_path, rxn_name, job_type='irc')
                except:
                    warnings.warn("Visualize sella IRC MEP failed.", UserWarning)

                # Perform minimization optimization on the forward and reverse directions' structures
                atoms1 = arrange_irc[0]
                atoms2 = arrange_irc[-1]
                atoms1.calc = calculator
                atoms2.calc = calculator
                opt1 = BFGS(atoms1, trajectory=os.path.join(rxn_output_path, f'{rxn_name}_opt1.traj'), logfile=os.path.join(rxn_output_path, f'{rxn_name}_opt1.log'))
                opt1.run(fmax=0.01, steps=steps)
                opt2 = BFGS(atoms2, trajectory=os.path.join(rxn_output_path, f'{rxn_name}_opt2.traj'), logfile=os.path.join(rxn_output_path, f'{rxn_name}_opt2.log'))
                opt2.run(fmax=0.01, steps=steps)
                # Read optimized energies of the two structures
                opt1_atoms = TrajectoryReader(os.path.join(rxn_output_path, f'{rxn_name}_opt1.traj'))[-1]
                opt2_atoms = TrajectoryReader(os.path.join(rxn_output_path, f'{rxn_name}_opt2.traj'))[-1]

                opt1_energy = opt1_atoms.get_potential_energy()
                opt2_energy = opt2_atoms.get_potential_energy()
                Ef = max([atoms.get_potential_energy() for atoms in arrange_irc]) - max(opt1_energy, opt2_energy)
                Er = max([atoms.get_potential_energy() for atoms in arrange_irc]) - min(opt1_energy, opt2_energy)
                # Save the reactant structures

                write(os.path.join(rxn_output_path, f'{rxn_name}_r.xyz'), opt1_atoms)
                write(os.path.join(rxn_output_path, f'{rxn_name}_p.xyz'), opt2_atoms)

                # # Calculate the reaction barrier values for the forward and reverse directions and store them in the run log(if )
                Ef = max([atoms.get_potential_energy() for atoms in arrange_irc]) - arrange_irc[0].get_potential_energy()
                Er = max([atoms.get_potential_energy() for atoms in arrange_irc]) - arrange_irc[-1].get_potential_energy()

                with open(os.path.join(rxn_output_path, 'run.log'), 'w') as f:
                    f.write(f'rxn_name: {rxn_name}\nref_time: {ref_time}\n'
                            f'refine_steps: {refine_steps}\nimaginary_count: {imaginary_count}\n'
                            f'irc_time: {irc_time}\n'f'irc_steps: {irc_steps}\n')
                    f.write('\nreaction barrier:\t' + f'Ef = {Ef} eV\tEr = {Er} eV\t\n')
                    f.write(f'\nirc split flag: {flag}\n')

                # Draw IRC optimization path in 3D structure
                nframes = len(irc_traj)
                optimized_sella = read(irc_file_dir, ":")[-1 * nframes:]
                # Make an interactive HTML file of the optimized neb trajectory
                x3d = X3D(optimized_sella, bond=True)
                x3d.write(os.path.join(rxn_output_path, f"optimized_sella_irc_{rxn_name}.html"))

                with open(os.path.join(output_path, 'all_log.txt'), 'a') as f:
                    f.write(f'{rxn_name}\t{ref_time}\t{refine_steps}\t{irc_time}\t{irc_steps}\t{Ef}\t{Er}\t{flag}\n')

                break

            except Exception as e:
                traceback.print_exc()
                if attempt <= max_retries:
                    print(f'Attempt {attempt} failed on {rxn_name} IRC. Retrying...')
                    time.sleep(3)  # Add an appropriate wait time
                    continue
                else:
                    print(f'IRC error on {rxn_name} after {max_retries + 1} attempts.')
                    continue

def main(args, DEVICE):
    input_path = args.input_path
    output_path = args.output_path + f'/{args.model_name}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the model and initialize the calculator
    try:
        if args.model_name == 'PaiNN':
            config = read_config(args.model_path + f'/{args.model_name}/config.txt')
            statedict = torch.load(args.model_path + f'/{args.model_name}/PaiNN.sd', map_location=DEVICE)
            new_statedict = {}
            for k, v in statedict.items():
                name = k[7:] if k.startswith('module.') else k
                new_statedict[name] = v
            model = PaiNN(int(config['DEFAULT']['num_interactions']), int(config['DEFAULT']['hidden_state_size']),
                          float(config['DEFAULT']['cutoff']))
            model.load_state_dict(new_statedict)
            model.eval()
            calculator = utils.MLCalculator(model)

        elif args.model_name == 'MACE':
            calculator = MACECalculator(model_paths=args.model_path + f'/{args.model_name}/MACE.model', device='cuda')

        elif args.model_name == 'MACE_deltaL':
            calculator1 = MACECalculator(model_paths=args.model_path + f'/{args.model_name}/MACE_deltaL.model', device='cuda')
            calculator2 = XTB(method='GFN2-xTB')
            calculator = SumCalculator([calculator1, calculator2]) 
        else:
            calculator = None
            raise ValueError(
                f"Unknown model name: {args.model_name}. Please choose from 'MACE_deltaL', 'MACE', 'PaiNN_nms'.")
    except Exception as e:
        print('An error was encountered while loading the model')

    rxn_names = [n for n in os.listdir(input_path) if n.endswith('.xyz')]
    rxn_names = natsorted(rxn_names, alg=ns.PATH)
    print(f'number of jobs: {len(rxn_names)}.')
    
    if calculator is not None:
        sella_refine_irc(rxn_names, input_path, output_path, calculator, redine_f_max=0.04, irc_f_max=0.1, steps=300)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--input_path", nargs="?", default=r'./example/input')
    parser.add_argument("--output_path", nargs="?", default=r'./example/output')
    parser.add_argument("--model_name", nargs="?", default='MACE_deltaL',
                        choices=['MACE', 'MACE_deltaL', 'PaiNN'],
                        help="Select the model name for transition state optimization. "
                             "Options include: 'MACE', 'MACE_deltaL', 'PaiNN'")
    parser.add_argument("--model_path", nargs="?", default=r'./models')
    args = parser.parse_args()

    main(args, DEVICE)
