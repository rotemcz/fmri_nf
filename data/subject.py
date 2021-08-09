from util.config import SubjectMetaData
from itertools import chain
from abc import abstractmethod
import torch
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import json
from dataclasses import dataclass, asdict


class Window:
    def __init__(self, idx, time_slots, window_type, bold_mat):
        self.idx = idx
        self.time = time_slots
        self.window_type = window_type
        self.bold = self.gen_bold_mat(bold_mat)

    @abstractmethod
    def gen_bold_mat(self, *args): pass

    @abstractmethod
    def get_data(self, width): pass

    @property
    def mean(self): return torch.mean(self.bold)

    @property
    def std(self): return torch.std(self.bold)


class Window3D(Window):
    def gen_bold_mat(self, bold_mat):
        voxels_md = Subject.voxels_md
        x = bold_mat[voxels_md.h_range, :, :, :]
        x = x[:, voxels_md.w_range, :, :]
        x = x[:, :, voxels_md.d_range, :]
        return torch.tensor(x[:, :, :, self.time])

    def get_data(self, width): return self.bold[:, :, :, :width]

    def __repr__(self): return f'{self.window_type}# {self.idx}: mean={self.mean}, std={self.std}'


class Window3DFull(Window):
    def gen_bold_mat(self, bold_mat):
        voxels_md = [Subject.voxels_md.h_range, Subject.voxels_md.w_range, Subject.voxels_md.d_range]
        voxels_range = [range(voxels_md[i][0]) if voxels_md[i][0] > bold_mat.shape[i] - voxels_md[i][-1] else \
          range(voxels_md[i][-1] + 1, bold_mat.shape[i]) for i in range(3)]
        x = bold_mat[voxels_range[0], :, :, :]
        x = x[:, voxels_range[1], :, :]
        x = x[:, :, voxels_range[2], :]
        return torch.tensor(x[:, :, :, self.time])

    def get_data(self, width): return self.bold[:, :, :, :width]

    def __repr__(self): return f'{self.window_type}# {self.idx}: mean={self.mean}, std={self.std}'

class PairedWindows:
    def __init__(self, amyg_window, full_brain_window):
        assert amyg_window.idx == full_brain_window.idx, f'indices mismatch: {amyg_window.idx} != {full_brain_window.idx}'
        self.idx = amyg_window.idx
        self.amyg_window: Window = amyg_window
        self.full_brain_window: Window = full_brain_window

    def __repr__(self):
        return f'Windows #{self.idx}'

    def get_data(self, width):
        res = torch.stack([w.get_data(width) for w in (self.amyg_window, self.full_brain_window)])
        return res


class Subject:
    voxels_md = None

    def __init__(self, regulate_times, bold_mat, subject_type, subject_name, window_data_type=Window3D):
        self.name = subject_name
        self.type_ = subject_type
        self.regulate_times = regulate_times
        amyg_data = [window_data_type(i, slot, 'regulate', bold_mat) for i, slot in enumerate(self.regulate_times)]
        full_scans = [Window3DFull(i, slot, 'full', bold_mat) for i, slot in enumerate(self.regulate_times)]
        self.paired_windows = [PairedWindows(amyg_data[i], full_scans[i]) for i in range(len(amyg_data))]
        self.indices_list = []


    def get_windows(self, windows_num):
        return self.paired_windows[:windows_num]

    def __len__(self):
        return len(self.paired_windows)

    @property
    def type(self):
        return 'healthy'

