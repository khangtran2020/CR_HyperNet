import argparse
import logging
import random

import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(no_cuda=False, gpus='0'):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")

def get_gaussian_noise(clipping_noise, noise_scale, sampling_prob, num_client, num_compromised_client=1):
    return (num_compromised_client*noise_scale*clipping_noise)/(sampling_prob*num_client)

def draw_noise_to_phi(hnet, num_draws, gaussian_noise):
    new_set_params = {}
    for key in hnet.state_dict():
        value = hnet.state_dict()[key]
        new_set_params[key] = torch.cat(num_draws * [value.view(tuple([1] + [x for x in value.size()]))])
        new_set_params[key] = new_set_params[key] + torch.normal(mean=0, std=gaussian_noise,size=new_set_params[key].size())
    return new_set_params