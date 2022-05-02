import os
import sys
path = "/".join([x for x in os.path.realpath(__file__).split('/')[:-2]])
sys.path.insert(0, path)
import argparse
import json
import logging
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import trange

import time

from Models.models import CNNHyper, CNNTarget
from Data.node_noniid import BaseNodes
from Utils.utils import get_device, set_logger, set_seed, str2bool
from copy import deepcopy


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


def train(data_name: str, data_path: str, classes_per_node: int, num_nodes: int,
          steps: int, inner_steps: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          seed: int, nclient_step: int) -> None:
    ###############################
    # init nodes, hnet, local net #
    ###############################
    nodes = BaseNodes(data_name, data_path, num_nodes, classes_per_node=classes_per_node,
                      batch_size=bs)

    embed_dim = embed_dim

    if embed_dim == -1:
        logging.info("auto embedding size")
        embed_dim = int(1 + num_nodes / 4)

    if data_name == "cifar10":
        hnet = CNNHyper(num_nodes, embed_dim, hidden_dim=hyper_hid, n_hidden=n_hidden, n_kernels=n_kernels)
        net = CNNTarget(n_kernels=n_kernels)
    elif data_name == "cifar100":
        hnet = CNNHyper(num_nodes, embed_dim, hidden_dim=hyper_hid,
                        n_hidden=n_hidden, n_kernels=n_kernels, out_dim=100)
        net = CNNTarget(n_kernels=n_kernels, out_dim=100)
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100']")

    hnet = hnet.to(device)
    net = net.to(device)

    ##################
    # init optimizer #
    ##################
    embed_lr = embed_lr if embed_lr is not None else lr
    optimizers = {
        'sgd': torch.optim.SGD(
            [
                {'params': [p for n, p in hnet.named_parameters() if 'embed' not in n]},
                {'params': [p for n, p in hnet.named_parameters() if 'embed' in n], 'lr': embed_lr}
            ], lr=lr  # , momentum=0.9, weight_decay=wd
        ),
        'adam': torch.optim.Adam(params=hnet.parameters(), lr=lr)
    }
    optimizer = optimizers[optim]
    criteria = torch.nn.CrossEntropyLoss()

    ################
    # init metrics #
    ################
    last_eval = -1
    best_step = -1
    best_acc = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    step_iter = trange(steps)
    print('steps', steps)
    print('step_iter', step_iter)
    print('save_path', save_path)

    results = defaultdict(list)

    name_add = 'train_batch_n' + str(num_nodes) + '_nc' + str(nclient_step) + '_lr' + str(lr) + '_ilr' + str(
        inner_lr) + '_seed' + str(seed) + '_noniid_c2'

    step_vect = []
    for step in step_iter:
        start_time = time.time()
        hnet.train()

        node_id_vect = random.sample(range(num_nodes), nclient_step)

        hnet_grads_all = defaultdict(list)
        c = 0
        for node_id in node_id_vect:
            # produce & load local network weights
            weights, _ = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
            net.load_state_dict(weights)

            # init inner optimizer
            inner_optim = torch.optim.SGD(
                net.parameters(), lr=inner_lr, momentum=.9, weight_decay=inner_wd
            )

            # storing theta_i for later calculating delta theta
            inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

            # inner updates -> obtaining theta_tilda
            for i in range(inner_steps):
                net.train()
                inner_optim.zero_grad()
                optimizer.zero_grad()

                batch = next(iter(nodes.train_loaders[node_id]))
                img, label = tuple(t.to(device) for t in batch)

                pred = net(img)

                loss = criteria(pred, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 50)

                inner_optim.step()

            optimizer.zero_grad()

            final_state = net.state_dict()
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

            hnet_grads_each = torch.autograd.grad(
                list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values())
            )
            hnet.embeddings.weight.grad = hnet_grads_each[0]
            optimizer.step()

            if c == 0:
                hnet_grads = deepcopy(list(hnet_grads_each))
                for t in range(len(hnet_grads)):
                    hnet_grads[t] = hnet_grads[t] / nclient_step
            else:
                tmp = list(hnet_grads_each)
                for t in range(len(hnet_grads)):
                    hnet_grads[t] += tmp[t] / nclient_step
            c += 1

        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g

        for n, p in hnet.named_parameters():
            if 'embed' in n:
                p.grad = None

        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        optimizer.step()

        if step % 10 == 0:
            last_eval = step
            step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, num_nodes, hnet, net, criteria, device,
                                                                  split="test")
            logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

            results['test_avg_loss'].append(avg_loss)
            results['test_avg_acc'].append(avg_acc)

            _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, num_nodes, hnet, net, criteria, device, split="val")
            if best_acc < val_avg_acc:
                best_acc = val_avg_acc
                best_step = step
                test_best_based_on_step = avg_acc
                test_best_min_based_on_step = np.min(all_acc)
                test_best_max_based_on_step = np.max(all_acc)
                test_best_std_based_on_step = np.std(all_acc)

            results['val_avg_loss'].append(val_avg_loss)
            results['val_avg_acc'].append(val_avg_acc)
            results['best_step'].append(best_step)
            results['best_val_acc'].append(best_acc)
            my_csv = pd.DataFrame(results)
            name_save = save_path + name_add + '.csv'
            my_csv.to_csv(name_save, index=False)
            with open(save_path + name_add + '.pt', 'wb') as f:  # bd0.5_cr0_double bd0.1_cr2
                torch.save([hnet, net], f)

        print('Finish one step in ', time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Hypernetwork with Lookahead experiment"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="cifar10", choices=['cifar10', 'cifar100'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")
    parser.add_argument("--num-nodes", type=int, default=100, help="number of simulated nodes")

    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--inner-steps", type=int, default=4, help="number of inner steps")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-5, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")
    parser.add_argument("--bt", type=int, default=10, help="batch")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="results/", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    if args.data_name == 'cifar10':
        args.classes_per_node = 2
    else:
        args.classes_per_node = 10

    train(
        data_name=args.data_name,
        data_path=args.data_path,
        classes_per_node=args.classes_per_node,
        num_nodes=args.num_nodes,
        steps=args.num_steps,
        inner_steps=args.inner_steps,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
        seed=args.seed,
        nclient_step=args.bt
    )
