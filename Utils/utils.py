import argparse
import collections
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

def draw_noise_to_phi(hnet, num_draws, gaussian_noise, device):
    new_set_params = {}
    for key in hnet.state_dict():
        value = hnet.state_dict()[key]
        new_set_params[key] = torch.cat(num_draws * [torch.unsqueeze(value, 0)])
        new_set_params[key] = new_set_params[key] + torch.normal(mean=0, std=gaussian_noise,size=new_set_params[key].size()).to(device)
    return new_set_params

def create_state_dict_at_one_draw(hnet, index, dict_of_state):
    new_set_params = []
    for key in hnet.state_dict():
        new_set_params.append((key, dict_of_state[key][index]))
    return collections.OrderedDict(new_set_params)

def float_to_binary(x, m, n):
    x_abs = np.abs(x)
    x_scaled = round(x_abs * 2 ** n)
    res = '{:0{}b}'.format(x_scaled, m + n)
    if x >= 0:
        res = '0' + res
    else:
        res = '1' + res
    return res

# binary to float
def binary_to_float(bstr, m, n):
    sign = bstr[0]
    bs = bstr[1:]
    res = int(bs, 2) / 2 ** n
    if int(sign) == 1:
        res = -1 * res
    return res

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def string_to_int(a):
    bit_str = "".join(x for x in a)
    return np.array(list(bit_str)).astype(int)

def join_string(a, num_bit, num_feat):
    res = []
    for i in range(num_feat):
        res.append("".join(str(x) for x in a[i*num_bit:(i+1)*num_bit]))
    return np.array(res)


def bit_rand(args, feat):
    num_data_point, num_feat = feat.shape
    eps_x = args.eps_ldp
    numbit = args.num_bit
    numbit_int = args.exponent_bit
    float_bin = lambda x: float_to_binary(x, numbit_int, numbit - numbit_int - 1)
    float_to_binary_vec = np.vectorize(float_bin)
    bin_float = lambda x: binary_to_float(x, numbit_int, numbit - numbit_int - 1)
    binary_to_float_vec = np.vectorize(bin_float)
    alpha = np.sqrt((eps_x + num_feat*numbit)/(2*num_feat*sum([np.exp(2*i*eps_x/numbit) for i in range(numbit)])))
    feat = float_to_binary_vec(feat)
    feat = np.apply_along_axis(string_to_int, axis=1, arr=feat)
    index_matrix = np.array(range(numbit))
    index_matrix = np.tile(index_matrix, (num_data_point, num_feat))
    p = 1 / (1 + alpha * np.exp(index_matrix * eps_x / numbit))
    del (index_matrix)
    p_temp = np.random.rand(p.shape[0], p.shape[1])
    perturb = (p_temp > p).astype(int)
    del (p)
    del (p_temp)
    perturb_feat = (perturb + feat) % 2
    del (perturb)
    del (feat)
    perturb_feat = np.apply_along_axis(join_string, axis=1, arr=perturb_feat)
    perturb_feat = binary_to_float_vec(perturb_feat)
    return perturb_feat


