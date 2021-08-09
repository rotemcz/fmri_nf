import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
import torch
import random


class E2EDataSet(Dataset):
    def __init__(self, load: Path, sub_to_md, sub_type):
        self.indices, self.subjects = self.load_subjects(load, sub_to_md, sub_type)
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    def dump(
        self
    ):
        dump_location=f'data/e2e_ds.pkl'
        with open(dump_location, 'wb') as file_:
            pickle.dump(self, file_)

    @staticmethod
    def load_subjects(data_path: Path, sub_to_md, sub_type):
      res = []
      subjects = {}
      type_dir = Path(data_path, sub_type)
      if type_dir.exists:
        for subject_path in type_dir.iterdir():
          subject: ValueFuncSubject = pickle.load(open(subject_path, 'rb'))
          if str(int(subject.name)) in sub_to_md:
            res.extend([(subject.name, indices, sub_type) for indices in subject.indices_list[:18]])
            subjects[subject.name] = (sub_type, subject)
      return res, subjects


    def __getitem__(self, item):
      name, indices, sub_type= self.indices[item]
      x0 = self.subjects[name][1].paired_windows[0].full_brain_window.bold[:,:,:,indices[0]]
      x1 = self.subjects[name][1].paired_windows[1].full_brain_window.bold[:,:,:,indices[1]]
      y0 = self.subjects[name][1].paired_windows[0].amyg_window.bold[:,:,:,indices[0]]
      y1 = self.subjects[name][1].paired_windows[1].amyg_window.bold[:,:,:,indices[1]]
      it = {'data': (x0, x1, y0),
              'gt': y1,
              'name': name,
              't': indices[0]}
      return it

    def __len__(self):
        return len(self.indices)

    def train_val_test_split(self, train_ratio, val_ratio):
      subjects = [x for x, (s_type, _) in self.subjects.items()]
      random.shuffle(subjects)
      train_len = int(len(subjects) * train_ratio)
      val_len = int(len(subjects) * val_ratio)

      self.train, self.val, self.test = \
        subjects[:train_len], subjects[train_len:(train_len + val_len)], subjects[(train_len + val_len):]

      train_idx = [i for i, x in enumerate(self.indices) if x[0] in self.train]
      self.train_ds = torch.utils.data.Subset(self, train_idx)
      val_idx = [i for i, x in enumerate(self.indices) if x[0] in self.val]
      self.val_ds = torch.utils.data.Subset(self, val_idx)
      test_idx = [i for i, x in enumerate(self.indices) if x[0] in self.test]
      self.test_ds = torch.utils.data.Subset(self, test_idx)
      return self.train_ds, self.val_ds, self.test_ds, self.subjects, self.train, self.val, self.test


class E2EDataModule(pl.LightningDataModule):

    def __init__(self, train_ratio, val_ratio, batch_size, sub_to_md, type_to_load=None, dataset_class=E2EDataSet, load=False):
        super().__init__()
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.dataset_class = dataset_class
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.ds: E2EDataSet = None
        self.type_to_load = type_to_load
        if load:
            self.load_ds()
        else:
            self.build_ds(sub_to_md)

    def build_ds(self, sub_to_md):

        ds = self.dataset_class(
            Path(r'data'), sub_to_md, self.type_to_load
        )

        self.train_ds, self.val_ds, self.test_ds, self.subjects, self.train, self.val, self.test = ds.train_val_test_split(self.train_ratio, self.val_ratio)
        self.ds = ds

    def load_ds(self):
        ds: ValueFuncDataSet = pickle.load(
            open(r'data/e2e_ds.pkl', 'rb')
        )

        self.ds = ds
        self.train_ds = ds.train_ds
        self.val_ds = ds.val_ds
        self.test_ds = ds.test_ds


    def train_dataloader(self, batch_size=None) -> DataLoader:
        if not batch_size:
            batch_size = self.batch_size
        return DataLoader(self.train_ds, batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1)

    def full_dataloader(self):
        return DataLoader(torch.utils.data.ConcatDataset([self.test_ds, self.val_ds, self.train_ds]), batch_size=1)


class E2EClassificationDataSet(Dataset):
    def __init__(self, sub_to_data, train_subs, val_subs, test_subs):
        self.data = list(sub_to_data.values())
        self.train_subs = train_subs
        self.val_subs = val_subs
        self.test_subs = test_subs
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    def dump(
        self
    ):
        dump_location=f'data/e2e_classification_ds.pkl'
        with open(dump_location, 'wb') as file_:
            pickle.dump(self, file_)


    def __getitem__(self, item):
      losses, clusters, md, name = self.data[item]
      it = {'data': torch.stack([torch.tensor(losses).float(), torch.tensor(clusters).float()]),
              'name': name,
              'md': md}
      return it

    def __len__(self):
        return len(self.data)

    def train_val_test_split(self, train_ratio, val_ratio):
      train_idx = [i for i, x in enumerate(self.data) if x[3] in self.train_subs]
      val_idx = [i for i, x in enumerate(self.data) if x[3] in self.val_subs]
      test_idx = [i for i, x in enumerate(self.data) if x[3] in self.test_subs]
      self.train_ds = torch.utils.data.Subset(self, train_idx)
      self.val_ds = torch.utils.data.Subset(self, val_idx)
      self.test_ds = torch.utils.data.Subset(self, test_idx)
      return self.train_ds, self.val_ds, self.test_ds


class E2EClassificationDataModule(pl.LightningDataModule):

    def __init__(self, train_ratio, val_ratio, batch_size, sub_to_data, train_subs, val_subs, test_subs, dataset_class=E2EClassificationDataSet, load=False):
        super().__init__()
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.dataset_class = dataset_class
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.ds: E2EClassificationDataSet = None

        if load:
            self.load_ds()
        else:
            self.build_ds(sub_to_data, train_subs, val_subs, test_subs)

    def build_ds(self, sub_to_data, train_subs, val_subs, test_subs):

        ds = self.dataset_class(
            sub_to_data, train_subs, val_subs, test_subs
        )

        self.train_ds, self.val_ds, self.test_ds = ds.train_val_test_split(self.train_ratio, self.val_ratio)
        self.ds = ds

    def load_ds(self):
        ds: ValueFuncDataSet = pickle.load(
            open(r'data/e2e_classification_ds.pkl', 'rb')
        )

        self.ds = ds
        self.train_ds = ds.train_ds
        self.val_ds = ds.val_ds
        self.test_ds = ds.test_ds


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, drop_last=True)

    def full_dataloader(self):
        return DataLoader(torch.utils.data.ConcatDataset([self.test_ds, self.val_ds, self.train_ds]), batch_size=1)
