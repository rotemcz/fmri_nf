from torch import nn
import torch.nn.functional as F
import torch
from itertools import chain
import pytorch_lightning as pl
import numpy as np


class FullBrainEmbed(pl.LightningModule):
    def __init__(self,
                 embedding_size,
                 lr,
                 use_embeddings: bool = True,
                 weight_decay=0,
                 n_subjects=None):
        super().__init__()

        #encoder
        #self.conv1_full = nn.Conv3d(2, 2, 3, stride=2)
        #self.conv2_full = nn.Conv3d(2, 1, 4, stride=3)
        #self.fc1_full = nn.Linear(648 + embedding_size, 64) if use_embeddings else nn.Linear(648, 64)
        self.conv1_full = nn.Conv3d(2, 3, 3, stride=2)
        self.conv2_full = nn.Conv3d(3, 1, 4, stride=3)
        self.fc1_full = nn.Linear(648, 256)
        self.fc2_full = nn.Linear(256, 64)

        #self.conv1_amyg = nn.Conv3d(1, 1, 2, stride=1)
        #self.fc1_amyg = nn.Linear(100 + embedding_size, 8) if use_embeddings else nn.Linear(100, 8)
        self.conv1_amyg = nn.Conv3d(1, 2, 2, stride=1)
        self.conv2_amyg = nn.Conv3d(2, 1, 3, stride=1)
        self.fc1_amyg = nn.Linear(18, 8)
        #self.fc2_amyg = nn.Linear(15, 8)

        #self.fc_encode1 = nn.Linear(8 + 64 + embedding_size, 64) if use_embeddings else \
        #  nn.Linear(8 + 64, 64)
        #self.fc_encode2 = nn.Linear(64, 18)
        #self.fc_encode1 = nn.Linear(8 + 64 + embedding_size, 2) if use_embeddings else \
        #  nn.Linear(8 + 64, 2)
        #self.fc_encode2 = nn.Linear(64, 2)
        self.fc_encode1 = nn.Linear(8 + 64, 18)
        #self.fc_decode = nn.Linear(2 + embedding_size, 18) if use_embeddings else nn.Linear(2, 18)

        #decoder
        self.deconv1 = nn.ConvTranspose3d(1, 1, 3, stride=1)
        self.deconv2 = nn.ConvTranspose3d(1, 1, 2, stride=1)

        self.embedding_size = embedding_size

        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        #if use_embeddings:
           # self.embedding_lut = nn.Embedding(n_subjects, embedding_size)
        self.use_embeddings = use_embeddings

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, data, subject_id, ground_truth):
        x0, x1, y0 = data
        #x0, x1, x2, y0, y1 = data
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

        #amyg = torch.flatten(F.relu(self.conv1_amyg(amyg.float())), start_dim=1)
        amyg = F.relu(self.conv1_amyg(amyg.float()))
        amyg = torch.flatten(self.conv2_amyg(amyg), start_dim=1)
        amyg = F.relu(self.fc1_amyg(amyg))
        ##amyg = self.fc2_amyg(amyg)

        x = torch.cat((full, amyg), 1)
        x = F.relu(self.fc_encode1(x))
        '''
        if self.use_embeddings:
          emb = self.embedding_lut(torch.tensor(subject_id)).squeeze(0)
          full = torch.cat((full, emb), 1)
          full = F.relu(self.fc1_full(full))
          amyg = torch.cat((amyg, emb), 1)
          amyg = F.relu(self.fc1_amyg(amyg))

          x = torch.cat((full, amyg, emb), 1)
          x = F.relu(self.fc_encode1(x))
          x = torch.cat((x, emb), 1)
          x = self.fc_decode(x)
        else:
          full = F.relu(self.fc1_full(full))
          amyg = F.relu(self.fc1_amyg(amyg))
          x = torch.cat((full, amyg), 1)
          x = self.fc_decode(F.relu(self.fc_encode1(x)))
        '''
        ##x = self.fc_decode(self.fc_encode2(F.relu(self.fc_encode1(x))))
        #x = self.fc_decode(F.relu(self.fc_encode1(x)))

        x = x.view([x.shape[0], 1, 3, 2, 3])

        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)

        #loss = self.loss_fn(x.squeeze(1), ground_truth.float())

        return x.squeeze(1), ground_truth.float()

    def training_step(self, batch, batch_idx):
        #loss = self(batch['data'], batch['idx'], batch['gt'])
        res, gt = self(batch['data'], 0, batch['gt'])
        loss = torch.mean(self.loss_fn(res, gt))

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self(batch['data'], batch['idx'], batch['gt'])
        #loss = torch.mean(loss)
        res, gt = self(batch['data'], 0, batch['gt'])
        loss = torch.mean(self.loss_fn(res, gt))

        self.log('val_loss', loss)
        return loss
    '''
    def test_step(self, batch, batch_idx):
        res, gt = self(batch['data'], batch['idx'], batch['gt'])
        clusters = batch['cluster']
        losses = [[],[],[],[],[]]
        print(res.shape[0])
        for i in range(res.shape[0]):
            curr = self.loss_fn(res[i], gt[i])
            losses[clusters[i]].append(curr)
        losses_mean = []
        for l in losses:
            losses_mean.append(torch.mean(l).item())
        #loss = torch.mean(loss)
        print('losses: ', losses_mean)
        self.log('test_loss', losses_mean)
        return losses
    '''

    def test_step(self, batch, batch_idx):
        res, gt = self(batch['data'], 0, batch['gt'])
        #batch_clusters = batch['cluster']
        loss = torch.mean(self.loss_fn(res, gt))
        #clusters = torch.zeros(res.shape[0])
        #for i in range(res.shape[0]):
        #    loss[i] = self.loss_fn(res[i], gt[i])
        #    clusters[i] = batch_clusters[i]
        self.log('test_loss', loss)
        #return {'loss': loss, 'clusters': clusters}
        return {'loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'logs': logs}
       # avg_losses = torch.zeros(5)
       # for i in range(5):
       #     avg_losses[i] = torch.mean(torch.FloatTensor([x['loss'] for x in outputs if x['clusters'] == i]))
       # print(f'****************************avg_losses: {avg_losses} *************************************************')
       # logs = {'test_loss': torch.mean(avg_losses)}
       # return {'avg_test_loss0': avg_losses[0], 'avg_test_loss1': avg_losses[1], 'avg_test_loss2': avg_losses[2], 'avg_test_loss3': avg_losses[3], 'avg_test_loss4': avg_losses[4], 'log': logs}
        


class ConditionClassifier(pl.LightningModule):
    def __init__(self,
                 lr,
                 weight_decay=0):
        super().__init__()

        #self.fc = nn.Linear(8, 1)
        #self.fc = nn.Linear(10, 1)
        self.fc = nn.Linear(5, 1)
        #self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, data):
        data = torch.flatten(data, start_dim=0)

        return self.fc(data)
        #return F.softmax(self.fc(data), dim=1)
        #return F.log_softmax(self.fc(data), dim=1)

    def training_step(self, batch, batch_idx):
        #res = self(batch['data'][0])
        data = batch['data'][0][0]
        res = self(batch['data'][0][1])
        #gt = torch.stack([batch['md']['age'].double(), batch['md']['past'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        #gt = batch['md']['age'].float()
        #gt = batch['md']['TAS1'].float()
        gt = batch['md'].float()
        #gt = batch['md']['CAPS'].float()
        #gt = batch['md']['STAI_S1'].float()
        loss = torch.mean(self.loss_fn(res, gt))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        #res = self(batch['data'][0])
        res = self(batch['data'][0][1])
        #gt = torch.stack([batch['md']['age'].double(), batch['md']['past'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        #gt = batch['md']['age'].float()
        gt = batch['md'].float()
        #gt = batch['md']['TAS1'].float()
        #gt = batch['md']['CAPS'].float()
        #gt = batch['md']['STAI_S1'].float()
        loss = torch.mean(self.loss_fn(res, gt))
        #loss = torch.mean(self.loss_fn(res, batch['class']))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        #res = self(batch['data'][0])
        res = self(batch['data'][0][1])
        #gt = batch['class']
        #print(f'predicted: {res}, true: {gt}')
        #loss = torch.mean(self.loss_fn(res, batch['class']))
        #correct = False
        #if torch.max(res) == res[0][gt]:
        #    correct = True
        #gt = torch.stack([batch['md']['age'].double(), batch['md']['past'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
       # gt = batch['md']['CAPS'].float()
        #gt = batch['md']['TAS1'].float()
        gt = batch['md'].float()
        #gt = batch['md']['STAI_S1'].float()
        #gt = batch['md']['age'].float()
        print(f'pred is: {res}, and gt is: {gt}')
        #age_loss = torch.mean(self.loss_fn(res[0][0], batch['md']['age']))
        loss = torch.mean(self.loss_fn(res, gt))
        #past_loss = torch.mean(self.loss_fn(res[0][1], batch['md']['past']))
        #tas1_loss = torch.mean(self.loss_fn(res[0][2], batch['md']['TAS1'].float()))
        #stai_loss = torch.mean(self.loss_fn(res[0][3], batch['md']['STAI_S1'].float()))
        return {'loss': loss}
        #loss = torch.mean(self.loss_fn(res, gt))
        #self.log('test_loss', loss)
        #return {'loss': loss, 'age_loss': age_loss, 'past_loss': past_loss, 'tas1_loss': tas1_loss, 'stai_loss': stai_loss}
        #return {'loss': loss, 'correct': correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #avg_age_loss = torch.stack([x['age_loss'] for x in outputs]).mean()
        #avg_past_loss = torch.stack([x['past_loss'] for x in outputs]).mean()
        #avg_tas1_loss = torch.stack([x['tas1_loss'] for x in outputs]).mean()
        #avg_stai_loss = torch.stack([x['stai_loss'] for x in outputs]).mean()
        #accuracy = sum([x['correct'] for x in outputs]) / len(outputs)
        #logs = {'test_loss': avg_loss, 'avg_age_loss': avg_age_loss, 'avg_past_loss': avg_past_loss, 'avg_tas1_loss': avg_tas1_loss, 'avg_stai_loss': avg_stai_loss}
        #logs = {'test_loss': avg_loss, 'test_accuracy': accuracy}
        #return {'avg_test_loss': avg_loss, 'log': logs}
        return {'avg_test_loss': avg_loss}


class BasicCNNClassifier(pl.LightningModule):
    def __init__(self,
                 lr,
                 weight_decay=0):
        super().__init__()

        #encoder
        self.conv1_full = nn.Conv3d(36, 3, 3, stride=2)
        self.conv2_full = nn.Conv3d(3, 1, 4, stride=3)
        self.fc1_full = nn.Linear(648, 64)

        self.conv_amyg = nn.Conv3d(36, 1, 3, stride=1)
        self.fc1_amyg = nn.Linear(48, 8)

        self.fc = nn.Linear(72, 2)

        self.loss_fn = nn.MSELoss(reduction='none')
        #self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, data):
        x0, x1, y0 = data[0]
        y1 = data[1]
        if len(x0.shape) == 3:
            x0 = torch.stack([x0], dim=0)
            x1 = torch.stack([x1], dim=0)
            y0 = torch.stack([y0], dim=0)
            y1 = torch.stack([y1], dim=0)
        full = torch.cat([x0, x1], dim=-1).permute(0,4,1,2,3)
        amyg = torch.cat([y0, y1], dim=-1).permute(0,4,1,2,3)
        #amyg = y0.permute(0,4,1,2,3)

        full = F.relu(self.conv1_full(full.float()))
        full = torch.flatten(self.conv2_full(full), start_dim=1)
        full = self.fc1_full(full)

        amyg = F.relu(self.conv_amyg(amyg.float()))
        amyg = torch.flatten(amyg, start_dim=1)
        amyg = F.relu(self.fc1_amyg(amyg))

        x = torch.cat((full, amyg), 1)
        #return F.log_softmax(self.fc(x), dim=1)
        #return F.softmax(self.fc(x), dim=1)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        #res = self(batch['data'])
        res = self((batch['data'],batch['gt']))
        #res = torch.stack([res]) 
        gt = torch.stack([batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        #gt = torch.stack([batch['md']['CAPS'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        #gt = torch.stack([batch['md']['age'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        
        #loss = torch.mean(self.loss_fn(res, gt))
        #loss = torch.mean(self.loss_fn(res, batch['class']))
        #gt = batch['md']['past']
        loss = torch.mean(self.loss_fn(res, gt))
        self.log('train_loss', torch.mean(loss))
        return torch.mean(loss)

    def validation_step(self, batch, batch_idx):
        #res = self(batch['data'])
        res = self((batch['data'],batch['gt']))
        #res = torch.stack([res]) 
        #gt = batch['md']['age'].float()

        gt = torch.stack([batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        #gt = torch.stack([batch['md']['CAPS'].double(), batch['md']['TAS1'].double(), batch['md']['STAI_S1'].double()], dim=1).float()
        #gt = torch.stack([batch['md']['age'].double(), batch['md']['TAS1'].double(), batch['md']['STAI_S1'].double()], dim=1).float()
        #gt = batch['md']['past']
        loss = torch.mean(self.loss_fn(res, gt))
        #loss = torch.mean(self.loss_fn(res, gt))
        #loss = torch.mean(self.loss_fn(res, batch['class']))
        self.log('val_loss', torch.mean(loss))
        return torch.mean(loss)


    def test_step(self, batch, batch_idx):
        res = self((batch['data'],batch['gt']))
        #gt = batch['md']['past']
        gt = torch.stack([batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        #gt = torch.stack([batch['md']['CAPS'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        #gt = torch.stack([batch['md']['age'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        #correct = False 
        #if torch.max(res) == res[0][gt]:
        #   correct = True

        #caps_loss = torch.mean(self.loss_fn(res[0][0], batch['md']['CAPS']))
        #age_loss = torch.mean(self.loss_fn(res[0][0], batch['md']['age']))
        #loss = torch.mean(self.loss_fn(res, gt))
        #past_loss = torch.mean(self.loss_fn(res[0][1], batch['md']['past']))
        tas1_loss = torch.mean(self.loss_fn(res[0][0], batch['md']['TAS1'].float()))
        stai_loss = torch.mean(self.loss_fn(res[0][1], batch['md']['STAI_S1'].float()))
        soft = F.softmax(res, dim=1)
        
        print(f'pred is: {soft}, res is: {res} gt is: {gt}')
        loss = torch.mean(self.loss_fn(res, gt))
        self.log('test_loss', torch.mean(loss))
        #return {'loss': torch.mean(loss), 'correct': correct}
        #return {'loss': torch.mean(loss)}
        return {'loss': torch.mean(loss), 'tas_loss': tas1_loss, 'stai_loss': stai_loss}
        #return {'loss': torch.mean(loss), 'caps_loss': caps_loss, 'tas_loss': tas1_loss, 'stai_loss': stai_loss}
        #return {'loss': torch.mean(loss), 'age_loss': age_loss, 'tas_loss': tas1_loss, 'stai_loss': stai_loss}
        
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #accuracy = sum([x['correct'] for x in outputs]) / len(outputs)
        #avg_caps_loss = torch.stack([x['caps_loss'] for x in outputs]).mean()
        #avg_age_loss = torch.stack([x['age_loss'] for x in outputs]).mean()
        avg_tas_loss = torch.stack([x['tas_loss'] for x in outputs]).mean()
        avg_stai_loss = torch.stack([x['stai_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss, 'avg_tas_loss': avg_tas_loss, 'avg_stai_loss': avg_stai_loss}
        #logs = {'test_loss': avg_loss, 'avg_tas_loss': avg_tas_loss, 'avg_stai_loss': avg_stai_loss, 'avg_caps_loss': avg_caps_loss}
        #logs = {'test_loss': avg_loss, 'avg_tas_loss': avg_tas_loss, 'avg_stai_loss': avg_stai_loss, 'avg_age_loss': avg_age_loss}
        
        #logs = {'test_loss': avg_loss, 'test_accuracy': accuracy}
        return {'avg_test_loss': avg_loss, 'log': logs}



class AmygHealthyClassifier(pl.LightningModule):
    def __init__(self,
                 lr,
                 weight_decay=0):
        super().__init__()

        #encoder
        self.conv1 = nn.Conv3d(18, 3, 3, stride=2)
        self.fc1 = nn.Linear(24, 3)

        self.loss_fn = nn.MSELoss(reduction='none')
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, data):
        y0, y1 = data
        x = y0 - y1
        if len(x.shape) == 3:
            x = torch.stack([x], dim=0)
        x = x.permute(0,4,1,2,3)

        x = F.relu(self.conv1(x.float()))
        x = torch.flatten(x, start_dim=1)

        return self.fc1(x)

    def training_step(self, batch, batch_idx):
        res = self(batch['data'])
        gt = torch.stack([batch['md']['age'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        loss = torch.mean(self.loss_fn(res, gt), dim=0)
        self.log('train_loss', torch.mean(loss))
        return torch.mean(loss)
        

    def validation_step(self, batch, batch_idx):
        res = self(batch['data'])
        gt = torch.stack([batch['md']['age'].double(), batch['md']['TAS1'], batch['md']['STAI_S1']], dim=1).float()
        loss = torch.mean(self.loss_fn(res, gt), dim=0)
        self.log('val_loss', torch.mean(loss))
        return torch.mean(loss)

    def test_step(self, batch, batch_idx):
        res = self(batch['data'])
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
        

class AmygPTSDClassifier(pl.LightningModule):
    def __init__(self,
                 lr,
                 weight_decay=0):
        super().__init__()

        #encoder
        self.conv1 = nn.Conv3d(18, 3, 3, stride=2)
        self.fc1 = nn.Linear(24, 3)

        self.loss_fn = nn.MSELoss(reduction='none')
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, data):
        y0, y1 = data
        x = y0 - y1
        if len(x.shape) == 3:
            x = torch.stack([x], dim=0)
        x = x.permute(0,4,1,2,3)

        x = F.relu(self.conv1(x.float()))
        x = torch.flatten(x, start_dim=1)

        return self.fc1(x)

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
        return {'loss': torch.mean(loss), 'caps_loss': loss[0], 'tas_loss': loss[1], 'stai_loss': loss[2]}
        
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_caps_loss = torch.stack([x['caps_loss'] for x in outputs]).mean()
        avg_tas_loss = torch.stack([x['tas_loss'] for x in outputs]).mean()
        avg_stai_loss = torch.stack([x['stai_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss, 'avg_caps_loss': avg_caps_loss, 'avg_tas_loss': avg_tas_loss, 'avg_stai_loss': avg_stai_loss}
        return {'avg_test_loss': avg_loss, 'log': logs}



class AmygFibroClassifier(pl.LightningModule):
    def __init__(self,
                 lr,
                 weight_decay=0):
        super().__init__()

        #encoder
        self.conv1 = nn.Conv3d(18, 3, 3, stride=2)
        self.fc1 = nn.Linear(24, 2)

        self.loss_fn = nn.MSELoss(reduction='none')
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, data):
        y0, y1 = data
        x = y0 - y1
        if len(x.shape) == 3:
            x = torch.stack([x], dim=0)
        x = x.permute(0,4,1,2,3)

        x = F.relu(self.conv1(x.float()))
        x = torch.flatten(x, start_dim=1)

        return self.fc1(x)

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
        self.log('test_loss', torch.mean(loss))
        return {'loss': torch.mean(loss), 'tas_loss': loss[0], 'stai_loss': loss[1]}
        
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_tas_loss = torch.stack([x['tas_loss'] for x in outputs]).mean()
        avg_stai_loss = torch.stack([x['stai_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss, 'avg_tas_loss': avg_tas_loss, 'avg_stai_loss': avg_stai_loss}
        return {'avg_test_loss': avg_loss, 'log': logs}



class FullBrainEmbedOneSession(pl.LightningModule):
    def __init__(self,
                 embedding_size,
                 lr,
                 use_embeddings: bool = True,
                 weight_decay=0,
                 n_subjects=None):
        super().__init__()

        #encoder
        self.conv1_full = nn.Conv3d(1, 3, 3, stride=2)
        self.conv2_full = nn.Conv3d(3, 1, 4, stride=3)
        self.fc1_full = nn.Linear(648, 256)
        self.fc2_full = nn.Linear(256, 64)


        self.fc_encode1 = nn.Linear(64, 18)

        #decoder
        self.deconv1 = nn.ConvTranspose3d(1, 1, 3, stride=1)
        self.deconv2 = nn.ConvTranspose3d(1, 1, 2, stride=1)

        self.embedding_size = embedding_size

        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_embeddings = use_embeddings

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, data, subject_id, ground_truth):
        x0, x1, y0 = data
        if len(x0.shape) == 3:
            x0 = torch.stack([x0], dim=0)
        full = torch.stack([x0], dim=1)

        full = F.relu(self.conv1_full(full.float()))
        full = torch.flatten(self.conv2_full(full), start_dim=1)
        full = F.relu(self.fc1_full(full))
        full = self.fc2_full(full)

        x = F.relu(self.fc_encode1(full))

        x = x.view([x.shape[0], 1, 3, 2, 3])

        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)

        #loss = self.loss_fn(x.squeeze(1), ground_truth.float())

        return x.squeeze(1), y0.float()

    def training_step(self, batch, batch_idx):
        #loss = self(batch['data'], batch['idx'], batch['gt'])
        res, gt = self(batch['data'], 0, batch['gt'])
        loss = torch.mean(self.loss_fn(res, gt))

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self(batch['data'], batch['idx'], batch['gt'])
        #loss = torch.mean(loss)
        res, gt = self(batch['data'], 0, batch['gt'])
        loss = torch.mean(self.loss_fn(res, gt))

        self.log('val_loss', loss)
        return loss
    '''
    def test_step(self, batch, batch_idx):
        res, gt = self(batch['data'], batch['idx'], batch['gt'])
        clusters = batch['cluster']
        losses = [[],[],[],[],[]]
        print(res.shape[0])
        for i in range(res.shape[0]):
            curr = self.loss_fn(res[i], gt[i])
            losses[clusters[i]].append(curr)
        losses_mean = []
        for l in losses:
            losses_mean.append(torch.mean(l).item())
        #loss = torch.mean(loss)
        print('losses: ', losses_mean)
        self.log('test_loss', losses_mean)
        return losses
    '''

    def test_step(self, batch, batch_idx):
        res, gt = self(batch['data'], 0, batch['gt'])
        #batch_clusters = batch['cluster']
        loss = torch.mean(self.loss_fn(res, gt))
        #clusters = torch.zeros(res.shape[0])
        #for i in range(res.shape[0]):
        #    loss[i] = self.loss_fn(res[i], gt[i])
        #    clusters[i] = batch_clusters[i]
        self.log('test_loss', loss)
        #return {'loss': loss, 'clusters': clusters}
        return {'loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'logs': logs}
       # avg_losses = torch.zeros(5)
       # for i in range(5):
       #     avg_losses[i] = torch.mean(torch.FloatTensor([x['loss'] for x in outputs if x['clusters'] == i]))
       # print(f'****************************avg_losses: {avg_losses} *************************************************')
       # logs = {'test_loss': torch.mean(avg_losses)}
       # return {'avg_test_loss0': avg_losses[0], 'avg_test_loss1': avg_losses[1], 'avg_test_loss2': avg_losses[2], 'avg_test_loss3': avg_losses[3], 'avg_test_loss4': avg_losses[4], 'log': logs}
        
