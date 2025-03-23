# Add to python path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# General imports
import argparse
from tqdm import tqdm

# Project imports
from unet.base.ini_parser import IniParser
from unet.model.vanilla_unet import Unet
from unet.base.callbacks import CheckpointCallback, SaveBestCallback
from unet.base.misc import prepare_dataloader, get_loss_fnc, get_optimizer

# Pytorch imports
import torch


# Use GPU if available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

# Parse arguments
parser = argparse.ArgumentParser(
                    prog='Unet Training',
                    description='Train script for Unet model')
parser.add_argument('config_file')


def validate(dataloader, model, epoch, enable_amp, callbacks=None):
    size = len(dataloader.dataset)
    loss_values = []

    # Validate model
    model.eval()
    with torch.no_grad():
        with tqdm(total=size, desc='Validation: ') as pbar:
            for batch, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)

                with torch.autocast(device_type=device, dtype=torch.float16, enabled=enable_amp):
                    # Predict and calc loss
                    pred = model(x)
                    loss = loss_fn()(pred, y)

                # Print
                loss_values.append(loss.item())
                pbar.update(dataloader.batch_size)
                pbar.set_postfix({'loss': sum(loss_values) / len(loss_values),
                                  'min': min(loss_values),
                                  'max': max(loss_values)})

        # Callbacks after each epoch
        if callbacks:
            for cb in callbacks:
                cb.run(epoch, sum(loss_values) / len(loss_values))


def train(dataloader, model, loss_fn, optimizer, epoch, scaler, enable_amp=True, callbacks=None):
    size = len(dataloader.dataset)
    loss_values = []

    # Train model
    model.train()
    with tqdm(total=size, desc='Train: ') as pbar:
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=enable_amp):
                # Predict and calc loss
                pred = model(x)
                loss = loss_fn()(pred, y)

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Print
            current_batch_size = x.size(dim=0)
            loss_values.append(loss.item())
            pbar.update(current_batch_size)
            pbar.set_postfix({'loss': sum(loss_values) / len(loss_values),
                              'min': min(loss_values),
                              'max': max(loss_values)})


if __name__ == '__main__':
    args = parser.parse_args()
    p = IniParser(args.config_file)

    # create model
    unet = Unet(p.model.input_dim, 1, p.model.conv_dim).to(p.training.device)

    # Prepare Dataloader
    shuffle = p.data.shuffle
    split = p.data.split_train_val
    batch_size = p.training.batch_size
    keep_in_memory = p.data.keep_in_memory
    img_size = (p.preprocess.img_h, p.preprocess.img_w)
    train_dataloader, val_dataloader = prepare_dataloader(p.data.path_train, img_size, batch_size, shuffle, split, keep_in_memory)

    # Define loss-function and optimizer
    loss_fn = get_loss_fnc(p.training.loss)
    optimizer = get_optimizer(p.training.optimizer)
    optimizer = optimizer(unet.parameters(), lr=p.training.learning_rate)

    # Mixed Precision for faster training
    scaler = torch.GradScaler(p.training.device, enabled=p.misc.enable_amp)

    # Callback
    callbacks = []
    if p.callbacks.checkpoints:
        callbacks.append(CheckpointCallback(optimizer, model=unet, path=p.data.path_train))
    if p.callbacks.save_best:
        callbacks.append(SaveBestCallback(model=unet, path=p.data.path_train))

    # Train & validate for provided number of epochs
    for ep in range(p.training.epochs):
        print(f'Epoche {ep}:')
        train(train_dataloader, unet, loss_fn, optimizer, ep, scaler, p.misc.enable_amp)
        validate(val_dataloader, unet, ep, p.misc.enable_amp, callbacks=callbacks)
        print('')
    print("Done!")


