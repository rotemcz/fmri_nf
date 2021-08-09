from torch import nn
import torch.nn.functional as F
import torch
from itertools import chain
import pytorch_lightning as pl
import numpy as np


class FullBrainEmbed(pl.LightningModule):
    def __init__(self,
                 lr,
                 weight_decay=0,
                 n_subjects=None):
        super().__init__()

        #encoder
        self.conv1_full = nn.Conv3d(2, 3, 3, stride=2)
        self.conv2_full = nn.Conv3d(3, 1, 4, stride=3)
        self.fc1_full = nn.Linear(648, 256)
        self.fc2_full = nn.Linear(256, 64)

        self.conv1_amyg = nn.Conv3d(1, 2, 2, stride=1)
        self.conv2_amyg = nn.Conv3d(2, 1, 3, stride=1)
        self.fc1_amyg = nn.Linear(18, 8)

        self.fc_encode1 = nn.Linear(8 + 64, 18)

        #decoder
        self.deconv1 = nn.ConvTranspose3d(1, 1, 3, stride=1)
        self.deconv2 = nn.ConvTranspose3d(1, 1, 2, stride=1)

        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, data, subject_id, ground_truth):
        x0, x1, y0 = data
        if len(x0.shape) == 3:
            x0 = torch.stack([x0], dim=0)
            x1 = torch.stack([x1], dim=0)
            y0 = torch.stack([y0], dim=0)
        full = torch.stack([x0, x1], dim=1)
        amyg = torch.stack([y0], dim=1)

        full = F.relu(self.conv1_full(full.float()))
        full = torch.flatten(self.conv2_full(full), start_dim=1)
        full = F.relu(self.fc1_full(full))
        full = self.fc2_full(full)

        amyg = F.relu(self.conv1_amyg(amyg.float()))
        amyg = torch.flatten(self.conv2_amyg(amyg), start_dim=1)
        amyg = F.relu(self.fc1_amyg(amyg))

        x = torch.cat((full, amyg), 1)
        x = F.relu(self.fc_encode1(x))

        x = x.view([x.shape[0], 1, 3, 2, 3])

        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)

        return x.squeeze(1), ground_truth.float()

    def training_step(self, batch, batch_idx):
        res, gt = self(batch['data'], 0, batch['gt'])
        loss = torch.mean(self.loss_fn(res, gt))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        res, gt = self(batch['data'], 0, batch['gt'])
        loss = torch.mean(self.loss_fn(res, gt))

        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        res, gt = self(batch['data'], 0, batch['gt'])
        loss = torch.mean(self.loss_fn(res, gt))
        self.log('test_loss', loss)
        return {'loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'logs': logs}



class HealthyConditionClassifier(pl.LightningModule):
    def __init__(self,
                 lr,
                 input_size=10,
                 weight_decay=0):
        super().__init__()
        self.fc = nn.Linear(input_size, 3)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.lr = lr
        self.input_size = input_size
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, data):
      data = torch.flatten(data, start_dim=0)
      return self.fc(data)

    def training_step(self, batch, batch_idx):
        res = self(batch['data'])
        res = torch.stack([res])
        gt = torch.stack([batch['md']['age'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        loss = torch.mean(self.loss_fn(res, gt), dim=0)
        self.log('train_loss', torch.mean(loss))
        return torch.mean(loss)

    def validation_step(self, batch, batch_idx):
        res = self(batch['data'])
        res = torch.stack([res])
        gt = torch.stack([batch['md']['age'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        loss = torch.mean(self.loss_fn(res, gt), dim=0)
        self.log('val_loss', torch.mean(loss))
        return torch.mean(loss)

    def test_step(self, batch, batch_idx):
        res = self(batch['data'])
        res = torch.stack([res])
        gt = torch.stack([batch['md']['age'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        loss = torch.mean(self.loss_fn(res, gt), dim=0)
        self.log('test_loss', torch.mean(loss))
        return {'loss': torch.mean(loss), 'age_loss': loss[0], 'tas_loss': loss[1], 'stai_loss': loss[2]}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_age_loss = torch.stack([x['age_loss'] for x in outputs]).mean()
        avg_tas_loss = torch.stack([x['tas_loss'] for x in outputs]).mean()
        avg_stai_loss = torch.stack([x['stai_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss, 'avg_age_loss': avg_age_loss, 'avg_tas_loss': avg_tas_loss, 'avg_stai_loss': avg_stai_loss}
        return {'avg_test_loss': avg_loss, 'log': logs}

class PTSDConditionClassifier(pl.LightningModule):
    def __init__(self,
                 lr,
                 input_size=10,
                 weight_decay=0):
        super().__init__()

        self.fc = nn.Linear(input_size, 3)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.lr = lr
        self.input_size = input_size
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, data):
      data = torch.flatten(data, start_dim=0)
      return self.fc(data)

    def training_step(self, batch, batch_idx):
        res = self(batch['data'])
        gt = torch.stack([batch['md']['CAPS'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        loss = torch.mean(self.loss_fn(res, gt), dim=0)
        self.log('train_loss', torch.mean(loss))
        return torch.mean(loss)

    def validation_step(self, batch, batch_idx):
        res = self(batch['data'])
        gt = torch.stack([batch['md']['CAPS'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        loss = torch.mean(self.loss_fn(res, gt), dim=0)
        self.log('val_loss', torch.mean(loss))
        return torch.mean(loss)

    def test_step(self, batch, batch_idx):
        res = self(batch['data'])
        gt = torch.stack([batch['md']['CAPS'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        loss = torch.mean(self.loss_fn(res, gt), dim=0)
        self.log('test_loss', torch.mean(loss))
        return {'loss': torch.mean(loss), 'CAPS_loss': loss[0], 'tas_loss': loss[1], 'stai_loss': loss[2]}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_CAPS_loss = torch.stack([x['CAPS_loss'] for x in outputs]).mean()
        avg_tas_loss = torch.stack([x['tas_loss'] for x in outputs]).mean()
        avg_stai_loss = torch.stack([x['stai_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss, 'avg_CAPS_loss': avg_CAPS_loss, 'avg_tas_loss': avg_tas_loss, 'avg_stai_loss': avg_stai_loss}
        return {'avg_test_loss': avg_loss, 'log': logs}


class FibroConditionClassifier(pl.LightningModule):
    def __init__(self,
                 lr,
                 input_size=10,
                 weight_decay=0):
        super().__init__()

        self.fc = nn.Linear(input_size, 2)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.lr = lr
        self.input_size = input_size
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, data):
      data = torch.flatten(data, start_dim=0)
      return self.fc(data)

    def training_step(self, batch, batch_idx):
        res = self(batch['data'])
        gt = torch.stack([batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        loss = torch.mean(self.loss_fn(res, gt), dim=0)
        self.log('train_loss', torch.mean(loss))
        return torch.mean(loss)

    def validation_step(self, batch, batch_idx):
        res = self(batch['data'])
        gt = torch.stack([batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        loss = torch.mean(self.loss_fn(res, gt), dim=0)
        self.log('val_loss', torch.mean(loss))
        return torch.mean(loss)

    def test_step(self, batch, batch_idx):
        res = self(batch['data'])
        gt = torch.stack([batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        loss = torch.mean(self.loss_fn(res, gt), dim=0)
        new_loss = torch.div(torch.abs(res-gt), gt)
        self.log('test_loss', torch.mean(loss))
        return {'loss': torch.mean(loss), 'tas_loss': loss[0], 'stai_loss': loss[1]}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_tas_loss = torch.stack([x['tas_loss'] for x in outputs]).mean()
        avg_stai_loss = torch.stack([x['stai_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss, 'avg_tas_loss': avg_tas_loss, 'avg_stai_loss': avg_stai_loss}
        return {'avg_test_loss': avg_loss, 'log': logs}
