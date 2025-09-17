"""
mace模型测试脚本，包含画图程序。含数据密度的图建议采用plot_pred_targ_parallel函数，有加速效果。
"""
import os
import ase
import torch
import subprocess
from ase.io import read
from fastkde import fastKDE
import numpy as np
from mace import data
from tqdm import tqdm
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rcParams
from statistics import mean
from sklearn.metrics import explained_variance_score, r2_score, median_absolute_error, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from mace.tools import torch_geometric, torch_tools, utils
from aseMolec import pltProps as pp
from aseMolec import extAtoms as ea
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter
from fastkde import fastKDE
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = '15'

def run_testing(savedir, config_name):

    command = [
    #evaluate the test set
    "python3", "./eval_configs.py",
    f"--configs=./datasets/nms/v3/{config_name}_test.xyz",
    f"--model=./models/nms/v3/{config_name}_stagetwo.model",
    f"--output=f'{savedir}/{config_name}_pred.xyz",
    "--default_dtype=float32",
    "--info_prefix=",
    "--device=cuda"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

def run_test(model_dir, test_dataset_dir, batch_size):

    torch_tools.set_default_dtype("float32")
    device = torch_tools.init_device("cuda:0")

    # Load model
    model = torch.load(f=model_dir, map_location=device)
    model = model.to(
        device
    )  # shouldn't be necessary but seems to help with CUDA problems

    for param in model.parameters():
        param.requires_grad = False

    # Load data and prepare input
    atoms_list = ase.io.read(test_dataset_dir, index=":")
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max)
            )
            for config in configs
        ],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    forces_collection = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), compute_stress=False)
        energies_list.append(torch_tools.to_numpy(output["energy"]))

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        forces_collection.append(forces[:-1])  # drop last as its empty

    energies = np.concatenate(energies_list, axis=0)
    forces_list = [
        forces for forces_list in forces_collection for forces in forces_list
    ]
    return energies, forces_list

def plot_RMSEs(figdir, db1, db2, labs):
    plt.figure(figsize=(8,4), dpi=100)
    plt.subplot(1,2,1)
    pp.plot_prop(ea.get_prop(db1, 'info', 'energy', peratom=False).flatten(), \
                 ea.get_prop(db2, 'info', 'energy', peratom=False).flatten(), \
                 title=r'Energy $(\rm eV)$ ', labs=labs, rel=True)
    plt.subplot(1,2,2)
    pp.plot_prop(np.concatenate(ea.get_prop(db1, 'arrays', 'forces')).flatten(), \
                 np.concatenate(ea.get_prop(db2, 'arrays', 'forces')).flatten(), \
                 title=r'Forces $\rm (eV/\AA)$ ', labs=labs, rel=True)
    plt.tight_layout()
    plt.savefig(figdir)
    return

def plot_pred_targ(targ, pred, save_path, name='forces'):
    x = np.loadtxt(save_path + pred)
    y = np.loadtxt(save_path + targ)

    lim_min = min(np.min(x), np.min(y))
    lim_max = max(np.max(x), np.max(y))

    lim_min -= (lim_max-lim_min) * 0.1
    lim_max += (lim_max-lim_min) * 0.1

    # 计算散点密度
    # xy = np.vstack([x, y])
    # z = stats.gaussian_kde(xy)(xy) # 最耗时
    # idx = z.argsort()
    # x, y, z = x[idx], y[idx], z[idx]

    # 使用FastKDE计算核密度估计，numPoints控制分辨率
    z = fastKDE.pdf(x, y).values
    # 对z进行排序以便颜色映射
    z = z.flatten()
    
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    MSE = mean_squared_error(x, y)
    RMSE = np.power(MSE, 0.5)
    corr, _ = pearsonr(x, y)
    R2 = corr**2
    MAE = mean_absolute_error(x, y)

    # config = {"font.family":'Times New Roman',"font.size": 15,"mathtext.fontset":'stix'}
    # config = {"font.size": 15, "mathtext.fontset": 'stix'}
    # for key, value in config.items():
    #     rcParams[key] = value

    units = {
            'energy': 'eV',
            'forces': r'eV/Å',
        }
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    scatter = ax.scatter(x, y, marker='o', c=z * 100, edgecolors=None, s=15, cmap='RdBu_r',  alpha=0.8)
    cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30, label='frequency')
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'black', lw=1)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # 根据名称调整轴格式
    if name == 'energy':
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    r = lim_max - lim_min
    plt.text(lim_max - r * 0.05, lim_min + r * 0.15, '$R^2=%.3f$' % R2, horizontalalignment='right', family='DajaVu Sans', fontsize=15)
    plt.text(lim_max - r * 0.05, lim_min + r * 0.1, '$MAE=%.3f$' % MAE, horizontalalignment='right', family='DajaVu Sans', fontsize=15)
    plt.text(lim_max - r * 0.05, lim_min + r * 0.05, '$RMSE=%.3f$' % RMSE, horizontalalignment='right', family='DajaVu Sans', fontsize=15)

    ax.set_title(name.upper(), fontsize=16)
    ax.set_xlabel('predicted %s (%s)' % (name, units[name]), fontsize=15)
    ax.set_ylabel('target %s (%s)' % (name, units[name]), fontsize=15)
    ax.legend(loc='upper left', frameon = False) # labels=['频次']
    fig.savefig(save_path + f"/{name}.jpg", dpi=500, format="jpg")
    plt.show()


def compute_kde_chunk(x_chunk, y_chunk):
    """ 计算每个数据块的核密度估计 """
    xy = np.vstack([x_chunk, y_chunk])
    kde = gaussian_kde(xy, bw_method='silverman')
    return kde(xy)

def plot_pred_targ_parallel(targ, pred, save_path, name='forces', num_jobs=100):
    # 加载数据
    x = np.loadtxt(save_path + pred)
    y = np.loadtxt(save_path + targ)

    # 确定x, y的最小和最大值
    lim_min = min(np.min(x), np.min(y))
    lim_max = max(np.max(x), np.max(y))
    lim_min -= (lim_max - lim_min) * 0.1
    lim_max += (lim_max - lim_min) * 0.1

    # 分块计算KDE
    chunk_size = len(x) // num_jobs
    x_chunks = [x[i:i + chunk_size] for i in range(0, len(x), chunk_size)]
    y_chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]

    # 使用Joblib并行计算
    results = Parallel(n_jobs=num_jobs)(delayed(compute_kde_chunk)(x_chunk, y_chunk)
                                        for x_chunk, y_chunk in zip(x_chunks, y_chunks))

    # 合并所有KDE结果
    z = np.concatenate(results)
    
    # 对z进行排序以便颜色映射
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    # -------------------- 关键：将 z 手动归一化到 [0,1] --------------------
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min) # 归一化到 [0,1]
    # -----------------------------------------------------------------------
    
    # 计算误差指标
    MSE = mean_squared_error(x, y)
    RMSE = np.sqrt(MSE)
    corr, _ = pearsonr(x, y)
    R2 = corr ** 2
    MAE = mean_absolute_error(x, y)

    # 绘图设置
    units = {
        'energy': 'eV',
        'forces': r'eV/Å',
    }
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    scatter = ax.scatter(x, y, marker='o', c=z_norm, edgecolors=None, s=15, cmap='RdBu_r', alpha=1, zorder=2)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'black', lw=1, zorder=1)
    
    cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30,
                        label='frequency')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # 根据名称调整轴格式
    if name == 'energy':
        formatter = plt.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    r = lim_max - lim_min
    plt.text(lim_max - r * 0.05, lim_min + r * 0.15, '$R^2=%.3f$' % R2, horizontalalignment='right',
             family='DajaVu Sans', fontsize=15)
    plt.text(lim_max - r * 0.05, lim_min + r * 0.1, '$MAE=%.3f$' % MAE, horizontalalignment='right',
             family='DajaVu Sans', fontsize=15)
    plt.text(lim_max - r * 0.05, lim_min + r * 0.05, '$RMSE=%.3f$' % RMSE, horizontalalignment='right',
             family='DajaVu Sans', fontsize=15)

    ax.set_title(name.upper(), fontsize=16)
    ax.set_xlabel('predicted %s (%s)' % (name, units[name]), fontsize=15)
    ax.set_ylabel('target %s (%s)' % (name, units[name]), fontsize=15)
    ax.legend(loc='upper left', frameon=False)

    # 保存图片
    fig.savefig(save_path + f"/{name}.jpg", dpi=500, format="jpg")
    plt.show()

def plot_scatter(e_true, e_pred, f_true, f_pred, save_path):
    """
    绘制散点图
    :param pred: 预测值
    :param targ: 真实值
    :param save_path: 保存路径
    :param name: 图名称
    :return:
    """

    # units = {
    #     'energy': 'eV',
    #     'forces': r'eV/Å',
    # }
    #
    # fig, ax_figs = plt.subplots(1, 2, figsize=(12, 6))
    #
    # for ax, pred, targ, key in zip(ax_figs, [e_pred, f_pred], [e_true, f_true], units.keys()):
    #     # name_pred = key + '_pred.txt'
    #     # name_targ = key + '_targ.txt'
    #     # np.savetxt(save_path + name_pred, pred)
    #     # np.savetxt(save_path + name_targ, targ)
    #
    #     mae = abs(pred - targ).mean()
    #     rmse = np.sqrt(mean_squared_error(targ, pred))
    #
    #     #ax.hexbin(pred, targ, mincnt=1)
    #     ax.scatter(pred, targ, color='red')
    #
    #     lim_min = min(np.min(pred), np.min(targ))
    #     lim_max = max(np.max(pred), np.max(targ))
    #
    #     lim_min -= (lim_max-lim_min) * 0.1
    #     lim_max += (lim_max-lim_min) * 0.1
    #
    #     ax.set_xlim(lim_min, lim_max)
    #     ax.set_ylim(lim_min, lim_max)
    #     ax.set_aspect('equal')
    #
    #     ax.plot((lim_min, lim_max),
    #             (lim_min, lim_max),
    #             color='#000000',  #
    #             zorder=-1,
    #             linewidth=0.5)
    #
    #     r2 = r2_score(pred, targ)
    #
    #     if key == 'energy':
    #         formatter = ticker.ScalarFormatter(useMathText=True)
    #         formatter.set_scientific(True)
    #         formatter.set_powerlimits((-1,1))
    #
    #         ax.xaxis.set_major_formatter(formatter)
    #         ax.yaxis.set_major_formatter(formatter)
    #     else:
    #         pass
    #
    #     ax.set_title(key.upper(), fontsize=14)
    #     ax.set_xlabel('predicted %s (%s)' % (key, units[key]), fontsize=10)
    #     ax.set_ylabel('target %s (%s)' % (key, units[key]), fontsize=10)
    #
    #     ax.text(0.1, 0.9, 'MAE: %.3f %s' % (mae, units[key]),
    #             transform=ax.transAxes, fontsize=14)
    #     ax.text(0.1, 0.8, 'RMSE: %.3f %s' % (rmse, units[key]),
    #             transform=ax.transAxes, fontsize=14)
    #     ax.text(0.1, 0.7, '$R^2=%.3f$' % r2,
    #             transform=ax.transAxes, fontsize=14)
    #
    #     # 快速绘制测试结果图
    #     fig.savefig(save_path + "test.jpg", dpi=400, format="jpg")
    #     plt.show()

    # 绘制包含了数据密度分布的测试结果图
    plot_pred_targ_parallel(targ='forces_targ.txt', pred='forces_pred.txt', save_path=save_path, name='forces')
    plot_pred_targ_parallel(targ='energy_targ.txt', pred='energy_pred.txt', save_path=save_path, name='energy')

def get_e_true(db):
    E = []
    for at in db:
        E.append(at.calc.results['energy'])
    return np.array(E)

def get_f_true(db):
    F = []
    for at in db:
        F.append(at.calc.results['forces'])
    F_array = np.vstack(F).flatten()
    return F_array


if __name__ == "__main__":

    # config_name  = 'final_Rh_all_nms-f'
    # fig_name = f'{config_name}_test.png'
    # savedir = f'./tests/{config_name}/'
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir, exist_ok=True)
    # model_dir = f"./models/nms/v3/{config_name}_stagetwo.model"
    # test_dataset_dir = f"./datasets/nms/v3/{config_name}_test.xyz"
    #
    # # run_testing(savedir, config_name)  # 生成测试集的预测结果xyz文件
    #
    # energy_pred, forces_pred = run_test(model_dir, test_dataset_dir, batch_size=40)
    # forces_pred = np.vstack(forces_pred).flatten()
    # np.savetxt(savedir + 'energy_pred.txt', energy_pred)
    # np.savetxt(savedir + 'forces_pred.txt', forces_pred)
    #
    # db1 = read(test_dataset_dir, ':')
    # e_true = get_e_true(db1)
    # f_true = get_f_true(db1)
    # # db2 = read(f'{savedir}/{config_name}_pred.xyz', ':')
    # # e_pred = ea.get_prop(db2, 'info', 'energy', peratom=False).flatten()
    # e_pred = np.loadtxt(savedir + 'energy_pred.txt')
    # # f_true = np.concatenate(ea.get_prop(db1, 'arrays', 'forces')).flatten()
    # # f_pred = np.concatenate(ea.get_prop(db2, 'arrays', 'forces')).flatten()
    # f_pred = np.loadtxt(savedir + 'forces_pred.txt')
    #
    # plot_scatter(e_true, e_pred, f_true, f_pred, save_path=savedir)


    config_name  = 'final_Rh_all_nms-f'
    fig_name = f'{config_name}_test.png'
    savedir = f'./tests/{config_name}/'
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
    # model_dir = f"./models/no_nms/{config_name}/{config_name}_stagetwo.model"
    test_dataset_dir = f"./datasets/nms/v3/{config_name}_test.xyz"

    # energy_pred, forces_pred = run_test(model_dir, test_dataset_dir, batch_size=40)
    # forces_pred = np.vstack(forces_pred).flatten()
    # np.savetxt(savedir + 'energy_pred.txt', energy_pred)
    # np.savetxt(savedir + 'forces_pred.txt', forces_pred)

    db1 = read(test_dataset_dir, ':')
    e_true = get_e_true(db1)
    f_true = get_f_true(db1)
    e_pred = np.loadtxt(savedir + 'energy_pred.txt')
    f_pred = np.loadtxt(savedir + 'forces_pred.txt')

    plot_scatter(e_true, e_pred, f_true, f_pred, save_path=savedir)


