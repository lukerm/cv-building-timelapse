import os
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

# from .train_utils import CustomImageDataset, get_image_transforms, UNet
from src.train.train_utils import CustomImageDataset, get_image_transforms, UNet


ROOT_DIR_EXPERIMENTS = os.path.expanduser('~/cv-building-timelapse/data/experiments')
BATCH_SIZE = 5


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1).to(device=device)
    bce_loss =  torch.nn.BCELoss(reduction='mean')
    # mse_loss = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create a dataset and dataloader
    kp = 'R1'
    input_transform, target_transform = get_image_transforms()

    train_dataset = CustomImageDataset(
        annotations_filename=f'image_paths_{kp}_clean.csv',
        img_rootdir=os.path.join(ROOT_DIR_EXPERIMENTS, 'train'),
        keypoint_label=kp,
        input_transform=input_transform,
        target_transform=target_transform,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = CustomImageDataset(
        annotations_filename=f'image_paths_{kp}_clean.csv',
        img_rootdir=os.path.join(ROOT_DIR_EXPERIMENTS, 'val'),
        keypoint_label=kp,
        input_transform=input_transform,
        target_transform=target_transform,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_cadence = 25

    N_EPOCHS = 10
    for e in range(N_EPOCHS):
        print(f'---------')
        print(f'EPOCH {e}')

        running_loss = 0.
        for b, (image, target) in enumerate(train_dataloader):

            if b % val_cadence == 0:
                model.eval()
                since_val_start = time.time()
                val_loss = 0.
                keep_max_metric_numer, keep_max_metric_denom = 0., 0.

                with torch.no_grad():
                    for vi, (val_image, val_target) in enumerate(val_dataloader):
                        val_prediction = model(val_image)
                        loss = bce_loss(val_target, val_prediction)
                        val_loss += loss.item() * len(val_image)

                        target_max = torch.amax(val_target.squeeze(1), dim=(1,2))
                        pred_max = torch.amax(val_prediction.squeeze(1), dim=(1,2))
                        keep_max_metric_numer += sum(target_max * pred_max)
                        keep_max_metric_denom += sum(target_max)  # counts only those with 1s

                val_loss = val_loss / len(val_dataloader.dataset)
                # max metrics keeps an eye on whether we're converging to a trivial 0 solution - it should stay near 1 ideally
                keep_max_metric = keep_max_metric_numer / keep_max_metric_denom
                time_elapsed = time.time() - since_val_start
                t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f'({t}):    VAL: batch {str(b + 1).zfill(3)} loss: {val_loss:.4f} max_metric: {keep_max_metric:.3f} (time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s)')
                model.train()

            optimizer.zero_grad()
            prediction = model(image)
            loss = bce_loss(prediction, target)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                print('WARNING: NaN/Inf loss detected!')

            report_cadence = 10
            if b % report_cadence == report_cadence - 1:
                last_loss = running_loss / report_cadence
                t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f'({t}):  batch {str(b+1).zfill(3)} loss: {last_loss:.4f}')
                running_loss = 0.
