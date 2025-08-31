import os
import time
import torch
import shutil
import warnings
import logging
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
from ase.optimize.bfgs import BFGS
from natsort import ns, natsorted

import configparser
import traceback

num_cpu_threads = '1'
os.environ['OMP_NUM_THREADS']=num_cpu_threads
os.environ['MKL_NUM_THREADS']=num_cpu_threads
os.environ['OPENBLAS_NUM_THREADS']=num_cpu_threads
os.environ['NUMEXPR_NUM_THREADS']=num_cpu_threads

#Using cuda with jax results in more VRAM usage and worse performance, so we'll make sure jax runs on CPU instead.
os.environ['JAX_PLATFORMS'] = 'cpu'




def ts_irc_pipeline(inputxyz, output_path, rxn_output_path, calculator,
                    redine_f_max=0.01, irc_f_max=0.01, steps=1000, retry=2):
    """
    过渡态优化 + IRC 路径 + 两端极小值优化的主流程
    自动记录日志与关键路径参数，支持重试机制。
    """

    rxn_name = os.path.splitext(os.path.basename(inputxyz))[0]

    # Setup logging
    logfile_path = os.path.join(rxn_output_path, f'{rxn_name}.log')
    logging.basicConfig(filename=logfile_path,
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    log = logging.getLogger()

    # 汇总日志表头
    all_log_file = os.path.join(output_path, 'all_log.txt')
    if not os.path.exists(all_log_file):
        with open(all_log_file, 'w') as f:
            f.write('name\tref_time\trefine_steps\tirc_time\tirc_steps\tEf\tEr\tflag\n')

    for attempt in range(1, retry + 1):
        try:
            # Step 1: TS Optimization
            log.info(f'[Step 1] TS optimization for {rxn_name}')
            atoms = read(inputxyz)
            atoms.calc = calculator
            atoms.info = {'charge': 0, 'spin': 1}

            ts_traj_path = os.path.join(rxn_output_path, f'{rxn_name}_ts.traj')
            ts_final_xyz = os.path.join(rxn_output_path, f'{rxn_name}_ts.xyz')

            t0 = time.time()
            dyn = Sella(atoms, trajectory=ts_traj_path, eta=2e-2, gamma=0.0001, delta0=0.02)
            dyn.run(fmax=redine_f_max, steps=steps)
            t1 = time.time()
            ref_time = round(t1 - t0, 2)

            traj = TrajectoryReader(ts_traj_path)
            refine_steps = len(traj) - 1
            ts_final = traj[-1]
            write(ts_final_xyz, ts_final)

            log.info(f'Finished TS optimization in {ref_time}s, steps: {refine_steps}')

            # Step 2: Frequency Check
            vib_dir = os.path.join(rxn_output_path, 'vib')
            if os.path.exists(vib_dir):
                shutil.rmtree(vib_dir)
            ts_final.info = {'charge': 0, 'spin': 1}
            ts_final.calc = calculator
            vib = Vibrations(ts_final, name=vib_dir)
            vib.run()
            freqs = vib.get_frequencies()
            imaginary_count = np.sum(np.imag(freqs) != 0)
            log.info(f'Imaginary frequencies: {imaginary_count}')
            vib.summary(log=os.path.join(rxn_output_path, f'{rxn_name}_vibration.log'))

            print(f' TS optimization successful for {rxn_name}!')

            if imaginary_count > 6:
                log.warning(f'Too many imaginary modes (>6). Retrying...')
                continue

            # Step 3: IRC
            irc_traj_path = os.path.join(rxn_output_path, f'{rxn_name}_irc.traj')
            ts_final.calc = calculator
            log.info(f'[Step 3] IRC calculation for {rxn_name}')
            irc = IRC(ts_final, trajectory=irc_traj_path, dx=0.1, eta=1e-4, gamma=0.1)
            t2 = time.time()
            irc.run(fmax=irc_f_max, steps=steps, direction='forward')
            split = len(TrajectoryReader(irc_traj_path))
            irc.run(fmax=irc_f_max, steps=steps, direction='reverse')
            t3 = time.time()
            irc_time = round(t3 - t2, 2)
            log.info(f'IRC completed in {irc_time}s. Split index: {split}')
            
            irc_traj = TrajectoryReader(irc_traj_path)
            irc_steps = len(irc_traj)
            
            # Step 4: Visualize IRC
            irc_all = [irc_traj[i] for i in range(split - 1, -1, -1)]
            irc_all.extend(irc_traj[split:])
            try:
                fit = fit_images(irc_all)
                fit.plot()
                plt.savefig(os.path.join(rxn_output_path, f'{rxn_name}_irc.png'), dpi=300)
                animation.write_animation(os.path.join(rxn_output_path, f'{rxn_name}_irc.gif'),
                                          images=irc_all, writer='pillow', interval=50)
                plt.close()
            except Exception as e:
                log.warning('IRC visualization failed.')
            
            # Step 5: Optimize IRC endpoints
            atoms1, atoms2 = irc_all[0], irc_all[-1]
            for i, atoms in enumerate([atoms1, atoms2], 1):
                atoms.calc = calculator
                opt = BFGS(atoms, trajectory=os.path.join(rxn_output_path, f'{rxn_name}_opt{i}.traj'))
                opt.run(fmax=0.01, steps=steps)
            
            opt1 = TrajectoryReader(os.path.join(rxn_output_path, f'{rxn_name}_opt1.traj'))[-1]
            opt2 = TrajectoryReader(os.path.join(rxn_output_path, f'{rxn_name}_opt2.traj'))[-1]
            e1 = opt1.get_potential_energy()
            e2 = opt2.get_potential_energy()
            
            Ef = max([a.get_potential_energy() for a in irc_all]) - max(e1, e2)
            Er = max([a.get_potential_energy() for a in irc_all]) - min(e1, e2)
            
            # Save lower energy product/reactant
            if e1 > e2:
                write(os.path.join(rxn_output_path, f'{rxn_name}_r.xyz'), opt1)
            else:
                write(os.path.join(rxn_output_path, f'{rxn_name}_r.xyz'), opt2)
            
            #Step 6: Write concise logs
            with open(os.path.join(rxn_output_path, 'run.log'), 'w') as f:
                f.write(f'ref_time: {ref_time}\n')
                f.write(f'refine_steps: {refine_steps}\n')
                f.write(f'irc_time: {irc_time}\n')
                f.write(f'irc_steps: {irc_steps}\n')
                f.write(f'Ef: {Ef:.6f} eV\n')
                f.write(f'Er: {Er:.6f} eV\n')
                f.write(f'irc_split_flag: {split}\n')
            
            with open(all_log_file, 'a') as f:
                f.write(f'{rxn_name}\t{ref_time}\t{refine_steps}\t{irc_time}\t{irc_steps}\t{Ef:.6f}\t{Er:.6f}\t{split}\n')
            
            log.info(f'>> {rxn_name} successfully finished.')
            return  # Success, exit loop

        except Exception as e:
            log.error(f'Exception during attempt {attempt} for {rxn_name}: {str(e)}')
            traceback.print_exc()
            if attempt < retry:
                log.warning(f'Retrying {rxn_name} (attempt {attempt + 1}/{retry})...')
                time.sleep(2)
            else:
                log.error(f'{rxn_name} failed after {retry} attempts.')
                return  # Final failure


if __name__ == '__main__':

    # 模型名称以及输入文件路径
    input_path = r'./example/input'
    output_path = r'./example/output'
    rxn_names = [n for n in os.listdir(input_path) if n.endswith('.xyz')]
    rxn_names = natsorted(rxn_names, alg=ns.PATH)
    print(f'number of jobs: {len(rxn_names)}.')


    calc1 = MACECalculator(model_paths='./model/DeepEST-OS.model', device='cuda')
    calc2 = XTB(method='GFN2-xTB')
    calculator = SumCalculator([calc1, calc2])

    for rx in rxn_names:
        rxn_name = rx.split('.')[0]
        # work folder
        rxn_output_path = os.path.join(output_path, rxn_name)
        if not os.path.exists(rxn_output_path):
            os.makedirs(rxn_output_path)
        else:
            print(f'{rxn_name} already exists, skip.')
            continue

        input_xyz = os.path.join(input_path, rx)
        ts_irc_pipeline(input_xyz, output_path, rxn_output_path, calculator, redine_f_max=0.0025, irc_f_max=0.005, steps=1000)