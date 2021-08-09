from argparse import ArgumentParser
from e2e.value_func_data_module import E2EDataModule, E2EClassificationDataModule
from e2e.value_func_nets import PTSDConditionClassifier, HealthyConditionClassifier, FibroConditionClassifier, FullBrainEmbed
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import json
from kmeans_pytorch import kmeans
import numpy as np
from torch import nn
import math
import pickle

def find_nearest_center(x, centers):
    min_dist = math.inf
    closest = -1
    for i, center in enumerate(centers):
        dist = torch.sum((x - center)**2)
        if dist < min_dist:
            min_dist = dist
            closest = i
    return closest, min_dist

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--classifier_lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--classifier_batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--classification_epochs', type=int)
    parser.add_argument('--type', type=str, default='healthy')
    gpus = torch.cuda.device_count()
    print(f'using {gpus} gpus!')
    args = parser.parse_args()
    train_ratio, val_ratio = 0.6, 0.2

    sub_to_md = pickle.load(open(f'data/sub_to_md_{args.type}.pkl', 'rb'))
    filename = 'e2e_net-{epoch:02d}-{val_loss:.5f}'
    checkpoint_callback = ModelCheckpoint(
        dirpath='saved_models/',
        filename=filename,
        save_top_k=5,
        monitor='val_loss',
        )

    for k in range(10):
      data: E2EDataModule = E2EDataModule(train_ratio, val_ratio, args.batch_size, sub_to_md, type_to_load=args.type, load=False)
      x0s = []
      item_to_idx = {}
      for i, item in enumerate(data.train_dataloader(1)):
        x0s.append(torch.flatten(item['data'][0]))
        item_to_idx[(item['name'][0], item['t'].item())] = i
      x0s = torch.stack(x0s)
      num_clusters = 5
      cluster_ids_x, cluster_centers = [], []
      while True:
        cluster_ids_x, cluster_centers = kmeans(
            X=x0s, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
        )
        centers_hist, _ = np.histogram(cluster_ids_x.numpy(), bins=num_clusters)
        rerun = False
        if np.min(centers_hist) < 10:
            rerun = True
        if not rerun:
          break
      items_to_cluster = {}
      for i, item in enumerate(data.train_dataloader(1)):
        items_to_cluster[(item['name'][0], item['t'].item())] = cluster_ids_x[item_to_idx[(item['name'][0], item['t'].item())]].item()
      for item in data.val_dataloader():
        c, _ = find_nearest_center(torch.flatten(item['data'][0]), cluster_centers)
        items_to_cluster[(item['name'][0], item['t'].item())] = c
      for item in data.test_dataloader():
        c, _ = find_nearest_center(torch.flatten(item['data'][0]), cluster_centers)
        items_to_cluster[(item['name'][0], item['t'].item())] = c

      model = FullBrainEmbed(lr=args.lr, n_subjects=len(data.subjects))
      params = {'batch_size': args.batch_size,
                'lr': args.lr}
      pred_trainer = Trainer(checkpoint_callback=checkpoint_callback,
                        gpus=gpus,
                        max_epochs=args.epochs)
      pred_trainer.fit(model, data)
      pred_trainer.test()
      model.eval()
      loss_func = nn.MSELoss()
      sub_to_clusters = {}
      sub_to_losses = {}
      losses = []
      clusters = []
      for sub in data.subjects:
        sub_to_clusters[sub] = [0,0,0,0,0]
        sub_to_losses[sub] = [0,0,0,0,0]
      for item in data.full_dataloader():
        x0 = item['data'][0].cuda()
        x1 = item['data'][1].cuda()
        y0 = item['data'][2].cuda()
        y1 = item['gt'].cuda()
        res, gt = model((x0,x1,y0), 0, y1)
        loss = loss_func(res, gt)

        losses.append(loss.item())
        clusters.append(items_to_cluster[(item['name'][0], item['t'].item())])
        sub_to_clusters[item['name'][0]][items_to_cluster[(item['name'][0], item['t'].item())]] += 1
        sub_to_losses[item['name'][0]][items_to_cluster[(item['name'][0], item['t'].item())]] += loss.item() / 18

      sub_to_data = {sub: (sub_to_losses[sub], sub_to_clusters[sub], sub_to_md[str(int(sub))], sub) for sub in sub_to_losses}

      classification_data = E2EClassificationDataModule(train_ratio, val_ratio, args.classifier_batch_size, sub_to_data, data.train, data.val, data.test)

      if args.type == 'PTSD':
          ptsd_model = PTSDConditionClassifier(lr=args.classifier_lr)
          ptsd_trainer = Trainer(checkpoint_callback=checkpoint_callback,
                            gpus=gpus,
                            max_epochs=args.classification_epochs)
          ptsd_trainer.fit(ptsd_model, classification_data)
          ptsd_trainer.test(ptsd_model)
      elif args.type == 'Fibro':
          fibro_model = FibroConditionClassifier(lr=args.classifier_lr)
          fibro_trainer = Trainer(checkpoint_callback=checkpoint_callback,
                            gpus=gpus,
                            max_epochs=args.classification_epochs)
          fibro_trainer.fit(fibro_model, classification_data)
          fibro_trainer.test(fibro_model)

      elif args.type == 'healthy':
          healthy_model = HealthyConditionClassifier(lr=args.classifier_lr)
          healthy_trainer = Trainer(checkpoint_callback=checkpoint_callback,
                            gpus=gpus,
                            max_epochs=args.classification_epochs)
          healthy_trainer.fit(healthy_model, classification_data)
          healthy_trainer.test(healthy_model)
