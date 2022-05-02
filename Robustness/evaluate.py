import os
import sys
path = "/".join([x for x in os.path.realpath(__file__).split('/')[:-2]])
sys.path.insert(0, path)
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import torch.utils.data
from tqdm import trange
import time
from Models.models import CNNHyper, CNNTarget
from Data.node_noniid import BaseNodes
from Utils.utils import get_device, set_logger, set_seed, str2bool, get_gaussian_noise
from copy import deepcopy
from robustness import *

def eval_model(nodes, num_nodes, hnet, net, criteria, device, split):
    curr_results = evaluate(nodes, num_nodes, hnet, net, criteria, device, split=split)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    avg_loss = np.mean([val['loss'] for val in curr_results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]

    return curr_results, avg_loss, avg_acc, all_acc


@torch.no_grad()
def evaluate(nodes: BaseNodes, num_nodes, hnet, net, criteria, device, split='test'):
    hnet.eval()
    results = defaultdict(lambda: defaultdict(list))

    for node_id in range(num_nodes):  # iterating over nodes

        running_loss, running_correct, running_samples = 0., 0., 0.
        if split == 'test':
            curr_data = nodes.test_loaders[node_id]
        elif split == 'val':
            curr_data = nodes.val_loaders[node_id]
        else:
            curr_data = nodes.train_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            img, label = tuple(t.to(device) for t in batch)

            weights, _ = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
            net.load_state_dict(weights)
            pred = net(img)
            running_loss += criteria(pred, label).item()
            running_correct += pred.argmax(1).eq(label).sum().item()
            running_samples += len(label)

        results[node_id]['loss'] = running_loss / (batch_count + 1)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples

    return results

def user_dp_draw_noise(args, hnet):


    pass