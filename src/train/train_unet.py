import copy
import os
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

# from .train_utils import CustomImageDataset, get_image_transforms, UNet
from src.train.train_utils import CustomImageDataset, get_image_transforms, UNet

CROP_SIZE = 256
ROOT_DIR_EXPERIMENTS = os.path.expanduser(f'~/cv-building-timelapse/data/experiments/{CROP_SIZE}')
BATCH_SIZE = 16
N_EPOCHS = 60

EXPERIMENT_ID = '001'
SAVE_PATH = os.path.expanduser(f'~/cv-building-timelapse/models/experiments/{CROP_SIZE}/{EXPERIMENT_ID}')
os.makedirs(SAVE_PATH, exist_ok=True)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1).to(device=device)
    bce_loss =  torch.nn.BCELoss(reduction='mean')
    # mse_loss = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

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

    train_loss_history = []
    val_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = np.inf
    last_val_improvement = 0

    for e in range(N_EPOCHS):
        epoch_line = f'---------EPOCH {e}---------'
        with open(os.path.join(SAVE_PATH, 'losses.txt'), 'a') as f:
            f.write(epoch_line + '\n')

        running_loss = 0.0
        for b, (image, target) in enumerate(train_dataloader):
            if (e == b == 0) or (b == len(train_dataloader) - 1):
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
                val_loss_history.append(val_loss)
                val_loss_line = f'({t}):    VAL: batch {str(b + 1).zfill(3)} loss: {val_loss:.6f} max_metric: {keep_max_metric:.3f} (time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s)'
                print(val_loss_line)
                with open(os.path.join(SAVE_PATH, 'losses.txt'), 'a') as f:
                    f.write(val_loss_line + '\n')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    last_val_improvement = 0

                    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(os.path.join(SAVE_PATH, 'best_epochs.txt'), 'a') as f:
                        f.write(f'({t}) epoch={e} {val_loss=:.6f}\n')
                    torch.save(model.state_dict(), os.path.join(SAVE_PATH, f'best.model.weights'))
                    torch.save(optimizer.state_dict(), os.path.join(SAVE_PATH, f'best.optimizer.weights'))
                    torch.save(scheduler.state_dict(), os.path.join(SAVE_PATH, f'best.scheduler.weights'))
                else:
                    last_val_improvement += 1

                if scheduler:
                    scheduler.step(val_loss)

                # Early stopping, monitoring validation loss: must be bigger than patience parameter
                if last_val_improvement > 5:
                    break

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
                train_loss_history.append(last_loss)
                t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                train_loss_line = f'({t}):  batch {str(b+1).zfill(3)} loss: {last_loss:.6f}'
                with open(os.path.join(SAVE_PATH, 'losses.txt'), 'a') as f:
                    f.write(train_loss_line + '\n')
                running_loss = 0.

        t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(os.path.join(SAVE_PATH, 'last_epochs.txt'), 'a') as f:
            f.write(f'({t}) epoch={e}\n')
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, f'last.model.weights'))
        torch.save(optimizer.state_dict(), os.path.join(SAVE_PATH, f'last.optimizer.weights'))
        torch.save(scheduler.state_dict(), os.path.join(SAVE_PATH, f'last.scheduler.weights'))

