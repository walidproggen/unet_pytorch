# General imports
import os
import cv2
import argparse
import numpy as np
from os.path import join

# Torch imports
import torch

# Project imports
from unet.base.ini_parser import IniParser
from unet.base.misc import prepare_dataloader
from unet.model.vanilla_unet import Unet


# Parse arguments
parser = argparse.ArgumentParser(
                    prog='Unet Testing',
                    description='Test script for Unet model')
parser.add_argument('config_file')


def tensor_to_numpy(tensor):
    img_numpy = tensor.numpy()[0, :, :, :]
    img_numpy = np.rollaxis(img_numpy, 0, 3)
    # If color, convert to gray
    if img_numpy.shape[-1] > 1:
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
    return img_numpy * 255


def get_model_params(path, weights='best'):
    if weights == 'best':
        path_weights = join(path, 'weights', 'best_weight.pth')
        return torch.load(path_weights, weights_only=True)
    else:
        path_weights = ''
        if weights == 'last':
            path_ckpts = join(path, 'checkpoints')
            files = [file for file in os.listdir(path_ckpts) if file.endswith('.pt')]
            if len(files) > 0:
                path_weights = join(path_ckpts, files[-1])
        else:
            path_ckpts = join(path, 'checkpoints')
            files = [file for file in os.listdir(path_ckpts) if file.endswith('.pt')]
            files = [file for file in files if weights in file]
            if len(files) == 1:
                path_weights = join(path_ckpts, files[0])

        # Load weights
        weights_dict = torch.load(path_weights, weights_only=True)
        return weights_dict['model_state']


def test(model, dataloader, save_as, path_output, device='cpu'):
    output = join(path_output, 'predictions')
    os.makedirs(output, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            pred = pred.cpu()
            y = y.cpu()
            x = x.cpu()

            # From tensor to normalized image
            pred_norm = tensor_to_numpy(pred)
            y_norm = tensor_to_numpy(y)
            x_norm = tensor_to_numpy(x)

            if save_as == 'side-by-side':
                img_h = cv2.hconcat([x_norm, y_norm, pred_norm])
                cv2.imwrite(join(output, str(i) + '.jpg'), img_h)
            elif save_as == 'masks-only':
                cv2.imwrite(join(output, str(i) + '.jpg'), pred_norm)
            else:
                raise ValueError(f'Unknown option in <test>.<save_as>: {save_as}')


if __name__ == '__main__':
    # Load config
    args = parser.parse_args()
    p = IniParser(args.config_file)

    # Prepare dataset
    test_dataloader = prepare_dataloader(p.data.path_test, (p.preprocess.img_h, p.preprocess.img_w))

    # Prepare model
    unet = Unet(p.model.input_dim, 1, p.model.conv_dim).to(p.training.device)
    unet.load_state_dict(get_model_params(p.data.path_train, p.test.weights))

    # Run test
    test(unet, test_dataloader, p.test.save_as, p.data.path_test, p.training.device)
