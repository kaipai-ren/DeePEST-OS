import os
import random
import ase.io
import numpy as np
import scipy
import torch
import ase.db
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from collections import defaultdict

from torch.utils.data.dataset import Subset

# 修改后可以输出Hessian的计算器版本
class MLCalculator(Calculator):
    def __init__(
        self,
        model,
        implemented_properties=None,
        device=None,
        **kwargs,
    ):
        if not implemented_properties:
            implemented_properties = ["energy", "energy_var", "forces", "forces_var"]
        self.implemented_properties = implemented_properties
        pin_memory = (device == 'cuda')

        self.batch_handler = BatchHandler(pin_memory=pin_memory)

        super().__init__(**kwargs)

        # self.atoms_converter = atoms_converter
        self.model = model
        self.device = device
        if device:
            model.to(device)

    def calculate(
        self, atoms=None, properties=None, system_changes=None
    ):  # pylint:disable=unused-argument
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        if not system_changes:
            system_changes = all_changes

        if not properties:
            properties = ["energy", "forces"]

        if self.calculation_required(atoms, properties):
            super().calculate(atoms)
            batch = self.batch_handler.get_batch(atoms)

            results = self.model(batch)
            energies = np.array(results["energy"].cpu().detach().numpy().squeeze(1), dtype='float64')
            forces = np.array(results["forces"].cpu().detach().numpy(), dtype='float64')

            for force, energy, atom in zip(forces, energies, atoms):
                atom.calc.results = {
                    "energy": energy.squeeze(),
                    "forces": force,
                }  # pylint:disable=attribute-defined-outside-init

                if "energy_var" in results:
                    atoms.calc.results["energy_var"] = results["energy_var"].item()
                if "forces_var" in results:
                    atoms.calc.results["forces_var"] = np.array(
                        results["forces_var"].cpu().squeeze().detach().numpy()
                    )

            for atom in atoms:
                atom.calc.atoms = atom.copy()

    # def get_hessian(self, atoms=None):
    #
    #     if isinstance(atoms, Atoms):
    #         atoms = [atoms]
    #     super().calculate(atoms)
    #     batch = self.batch_handler.get_batch(atoms)
    #     results = self.model(batch, compute_hessian=True)
    #     hessian = np.array(results["hessian"].cpu().detach().numpy(), dtype='float64')
    #     if "hessian_var" in results:
    #         atoms.calc.results["hessian_var"] = np.array(
    #             results["hessian_var"].cpu().squeeze().detach().numpy()
    #         )
    #     return hessian


def batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def get_dataset(db, energy_key="energy", forces_key="forces"):
    dataset = AseDbData(
        db,
        TransformRowToGraphXyz(
            cutoff=5.0,
            energy_property=energy_key,
            forces_property=forces_key,
        ),
    )
    return dataset


class DummyRow():
    def __init__(self, atoms):
        self.atoms = atoms

    def toatoms(self):
        return self.atoms


class BatchHandler:
    def __init__(self, pin_memory):
        self.transform = TransformRowToGraphXyz()
        self.collate_atomsdata = CollateAtoms(pin_memory)

    def get_batch(self, atoms):
        dummyrows = [DummyRow(atom) for atom in atoms]
        graphdata = [self.transform(row) for row in dummyrows]
        return self.collate_atomsdata(graphdata)


class CollateAtoms:
    def __init__(self, pin_memory):
        self.pin_memory = pin_memory

    def __call__(self, graphs):
        dict_of_lists = {k: [dic[k] for dic in graphs] for k in graphs[0]}
        if self.pin_memory:
            def pin(x):
                if hasattr(x, "pin_memory"):
                    return x.pin_memory()
                return x
        else:
            pin = lambda x: x

        collated = {k: pin(pad_and_stack(dict_of_lists[k])) for k in dict_of_lists}
        return collated


def pad_and_stack(tensors):
    """Pad list of tensors if tensors are arrays and stack if they are scalars"""
    if tensors[0].shape:
        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=0
        )
    return torch.stack(tensors)


class AseDbData(torch.utils.data.Dataset):
    def __init__(self, asedb_path, transformer, **kwargs):
        super().__init__(**kwargs)

        self.asedb_path = asedb_path
        self.asedb_connection = ase.db.connect(asedb_path)
        self.transformer = transformer

    def __len__(self):
        return len(self.asedb_connection)

    def __getitem__(self, key):
        # Note that ASE databases are 1-indexed
        try:
            return self.transformer(self.asedb_connection[key + 1])
        except KeyError:
            raise IndexError("index out of range") # pylint: disable=raise-missing-from


class TransformRowToGraphXyz:
    """
    Transform ASE DB row to graph while keeping the xyz positions of the vertices

    """

    def __init__(
        self,
        cutoff=5.0,
        energy_property="energy",
        forces_property="forces",
        energy_reference_property=None,
    ):
        self.cutoff = cutoff
        self.energy_property = energy_property
        self.forces_property = forces_property
        self.energy_reference_property = energy_reference_property

    def __call__(self, row):
        atoms = row.toatoms()

        edges, edges_displacement = self.get_edges(atoms)

        # Extract energy and forces if they exists
        try:
            energy = np.copy([np.squeeze(row.data[self.energy_property])])
        except (KeyError, AttributeError):
            energy = np.zeros(len(atoms))
        try:
            forces = np.copy(row.data[self.forces_property])
        except (KeyError, AttributeError):
            forces = np.zeros((len(atoms), 3))
        default_type = torch.get_default_dtype()

        # pylint: disable=E1102
        graph_data = {
            "nodes": torch.tensor(atoms.get_atomic_numbers()),
            "nodes_xyz": torch.tensor(atoms.get_positions(), dtype=default_type),
            "num_nodes": torch.tensor(len(atoms.get_atomic_numbers())),
            "edges": torch.tensor(edges),
            "edges_displacement": torch.tensor(edges_displacement, dtype=default_type),
            "cell": torch.tensor(np.array(atoms.get_cell()), dtype=default_type),
            "num_edges": torch.tensor(edges.shape[0]),
            "energy": torch.tensor(energy, dtype=default_type),
            "forces": torch.tensor(forces, dtype=default_type),
        }

        return graph_data

    def get_edges(self, atoms):
        # Compute distance matrix
        pos = atoms.get_positions()
        dist_mat = scipy.spatial.distance_matrix(pos, pos)

        # Build array with edges and edge features (distances)
        valid_indices_bool = dist_mat < self.cutoff
        np.fill_diagonal(valid_indices_bool, False)  # Remove self-loops
        edges = np.argwhere(valid_indices_bool)  # num_edges x 2
        edges_displacement = np.zeros((edges.shape[0], 3))

        return edges, edges_displacement

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='../checkpoint/checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


__all__ = ["ReduceLROnPlateau"]
class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Extends PyTorch ReduceLROnPlateau by exponential smoothing of the monitored metric

    """

    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
        smoothing_factor=0.0,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            mode (str): One of `min`, `max`. In `min` mode, lr will
                be reduced when the quantity monitored has stopped
                decreasing; in `max` mode it will be reduced when the
                quantity monitored has stopped increasing. Default: 'min'.
            factor (float): Factor by which the learning rate will be
                reduced. new_lr = lr * factor. Default: 0.1.
            patience (int): Number of epochs with no improvement after
                which learning rate will be reduced. For example, if
                `patience = 2`, then we will ignore the first 2 epochs
                with no improvement, and will only decrease the LR after the
                3rd epoch if the loss still hasn't improved then.
                Default: 10.
            threshold (float): Threshold for measuring the new optimum,
                to only focus on significant changes. Default: 1e-4.
            threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
                dynamic_threshold = best * ( 1 + threshold ) in 'max'
                mode or best * ( 1 - threshold ) in `min` mode.
                In `abs` mode, dynamic_threshold = best + threshold in
                `max` mode or best - threshold in `min` mode. Default: 'rel'.
            cooldown (int): Number of epochs to wait before resuming
                normal operation after lr has been reduced. Default: 0.
            min_lr (float or list): A scalar or a list of scalars. A
                lower bound on the learning rate of all param groups
                or each group respectively. Default: 0.
            eps (float): Minimal decay applied to lr. If the difference
                between new and old lr is smaller than eps, the update is
                ignored. Default: 1e-8.
            verbose (bool): If ``True``, prints a message to stdout for
                each update. Default: ``False``.
            smoothing_factor: smoothing_factor of exponential moving average
        """
        super().__init__(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose,
        )
        self.smoothing_factor = smoothing_factor
        self.ema_loss = None

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if self.ema_loss is None:
            self.ema_loss = current
        else:
            self.ema_loss = (
                self.smoothing_factor * self.ema_loss
                + (1.0 - self.smoothing_factor) * current
            )
        super().step(current, epoch)

def random_state(seed_value):
    # random seed setting to ensure that the experimental results can be reproduced
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True

def split_data(data: AseDbData,
               split_type: str = 'random_with_repeated_tags',
               sizes: tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               ) -> tuple[Subset,
                        Subset,
                        Subset]:

    if split_type == 'random_with_repeated_tags': # Use to constrain data with the same smiles go in the same split.
        ts_idx_dict=defaultdict(set) # 新建一个以set为默认value的字典
        for i in range(len(data)):
            ts_idx_dict[data.asedb_connection[i+1].structure_idx].add(i) # 标签与序号的键值对
        index_sets=list(ts_idx_dict.values())
        random_state(seed)
        random.shuffle(index_sets)
        train,val,test=[],[],[]
        train_size = int(sizes[0] * len(data))
        val_size = int(sizes[1] * len(data))
        # 获取train,val,test的数据标签，即其在dataset中的索引
        for index_set in index_sets:
            if len(train)+len(index_set) <= train_size:
                train += index_set
            elif len(val) + len(index_set) <= val_size:
                val += index_set
            else:
                test += index_set

        return Subset(data,train), Subset(data,val), Subset(data,test) # 返回torch.utils.data.dataset.Subset 类型的数据集
