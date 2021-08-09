import torch
import pickle
import random
import math
import numpy as np
from util.config import ROIData
import nibabel as nib
from data.subject import Subject
from pathlib import Path
from argparse import ArgumentParser


def calc_roi():
    """Load Amygdala ROI matrix, and calculate real voxels and cubic range in each dimension"""
    img = nib.load(f'RAmyg.nii')
    roi = np.where(np.array(img.dataobj))
    amyg_vox = [vox for vox in zip(*roi)]
    min_sizes = map(min, roi)
    max_sizes = map(max, roi)
    h, w, d = list(map(lambda small, big: list(range(small, big + 1)), min_sizes, max_sizes))

    return ROIData(amyg_vox, h, w, d)


def find_nearest_scan(scan, bold_mat):
  min_dist = math.inf
  min_dist_idx = -1
  for i in range(bold_mat.shape[-1]):
    curr_dist = torch.sum(((bold_mat[:,:,:,i] - scan)**2).reshape(scan.shape[0] * scan.shape[1] * scan.shape[2]))
    if curr_dist < min_dist:
      min_dist = curr_dist
      min_dist_idx = i
  return min_dist_idx


if __name__ == '__main__':

  parser = ArgumentParser()

  parser.add_argument('--n_subjects', type=int)
  args = parser.parse_args()



  roi_path = Path('roi_dict.pkl')
  if roi_path.exists():
    roi_dict = pickle.load(open(str(roi_path), 'rb'))
  else:
    roi_dict = calc_roi()
    pickle.dump(roi_dict, open('roi_dict.pkl', 'wb'))
  Subject.voxels_md = roi_dict
  sub_to_md = {}
  regulate_times = [list(range(18)), list(range(18, 36))]
  for i in range(args.n_subjects):
    bold_mat = np.random.rand(91, 109, 91, 36)
    sub = Subject(regulate_times, bold_mat, 'healthy', str(i))
    indices_list = []
    for j in range(sub.paired_windows[0].full_brain_window.bold.shape[-1]):
      x0 = sub.paired_windows[0].full_brain_window.bold[:,:,:,j]
      idx_1 = find_nearest_scan(x0, sub.paired_windows[1].full_brain_window.bold)
      indices_list.append((j, idx_1))
    sub.indices_list = indices_list

    pickle.dump(sub, open(f'data/healthy/sub_{i}.pkl', 'wb'))
    sub_to_md[str(i)] = {'age': random.randint(15, 80), 'TAS1': random.random() * 100, 'STAI_S1': random.random() * 100}

  pickle.dump(sub_to_md, open(f'data/sub_to_md_healthy.pkl', 'wb'))
