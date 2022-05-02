import sys
import os
import sys

path = "/".join([x for x in os.path.realpath(__file__).split('/')[:-2]])
sys.path.insert(0, path)
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
from config import parse_args
from Utils.utils import get_device, set_logger, set_seed, get_gaussian_noise
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


def train(args, device) -> None:
    # training params
    nodes = BaseNodes(args.dataset, args.data_path, args.num_client, classes_per_node=args.classes_per_node,
                      batch_size=args.batch_size)
    sampling_prob = args.bt / args.num_client
    embed_dim = args.embed_dim

    if embed_dim == -1:
        logging.info("auto embedding size")
        embed_dim = int(1 + args.num_client / 4)

    if args.dataset == "cifar10":
        hnet = CNNHyper(args.num_client, embed_dim, hidden_dim=args.hyper_hid, n_hidden=args.n_hidden,
                        n_kernels=args.nkernels)
        net = CNNTarget(n_kernels=args.nkernels)
    elif args.dataset == "cifar100":
        hnet = CNNHyper(args.num_client, embed_dim, hidden_dim=args.hyper_hid,
                        n_hidden=args.n_hidden, n_kernels=args.nkernels, out_dim=100)
        net = CNNTarget(n_kernels=args.nkernels, out_dim=100)
    else:
        raise ValueError("choose dataset from ['cifar10', 'cifar100']")

    hnet = hnet.to(device)
    net = net.to(device)

    ##################
    # init optimizer #
    ##################
    embed_lr = args.embed_lr if args.embed_lr is not None else args.lr
    optimizers = {
        'sgd': torch.optim.SGD(
            [
                {'params': [p for n, p in hnet.named_parameters() if 'embed' not in n]},
                {'params': [p for n, p in hnet.named_parameters() if 'embed' in n], 'lr': embed_lr}
            ], lr=args.lr  # , momentum=0.9, weight_decay=wd
        ),
        'adam': torch.optim.Adam(params=hnet.parameters(), lr=args.lr)
    }
    optimizer = optimizers[args.optim]
    criteria = torch.nn.CrossEntropyLoss()

    ################
    # init metrics #
    ################
    last_eval = -1
    best_step = -1
    best_acc = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    step_iter = trange(args.num_steps)
    print('steps', args.num_steps)
    print('step_iter', step_iter)
    print('save_path', args.save_path)

    results = defaultdict(list)

    name_add = 'train_batch_n' + str(args.num_client) + '_nc' + str(args.bt) + '_lr' + str(args.lr) + '_ilr' + str(
        args.inner_lr) + '_seed' + str(args.seed) + '_noniid_c2'

    noise_std = get_gaussian_noise(clipping_noise=args.grad_clip, noise_scale=args.noise_scale,
                                   sampling_prob=sampling_prob, num_client=args.num_comp_cli)
    step_vect = []
    for step in step_iter:
        start_time = time.time()
        hnet.train()

        node_id_vect = random.sample(range(args.num_client), args.bt)

        hnet_grads_all = defaultdict(list)
        c = 0
        for node_id in node_id_vect:
            # produce & load local network weights
            weights, _ = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
            net.load_state_dict(weights)
            temp_net = deepcopy(hnet)
            # init inner optimizer
            inner_optim = torch.optim.SGD(
                net.parameters(), lr=args.inner_lr, momentum=.9, weight_decay=args.inner_wd
            )

            # storing theta_i for later calculating delta theta
            inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

            # inner updates -> obtaining theta_tilda
            for i in range(args.inner_steps):
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
            temp_net_list_grad = []
            for p, g in zip(temp_net.parameters(), hnet_grads_each):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(temp_net.parameters(), args.grad_clip)
            temp_net_list_grad = [p.grad for p in temp_net.parameters()]

            if c == 0:
                hnet_grads = deepcopy(temp_net_list_grad)
                for t in range(len(hnet_grads)):
                    hnet_grads[t] = hnet_grads[t] / args.bt
            else:
                tmp = temp_net_list_grad
                for t in range(len(hnet_grads)):
                    hnet_grads[t] += tmp[t] / args.bt
            c += 1

        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g

        # for n, p in hnet.named_parameters():
        #     if 'embed' in n:
        #         p.grad = None

        # print("",hnet.parameters().grad)

        for p in hnet.parameters():
            p.grad = p.grad + torch.normal(0, noise_std, p.grad.size()).to(device)/args.bt
        optimizer.step()

        if step % 10 == 0:
            last_eval = step
            step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, args.num_client, hnet, net, criteria, device,
                                                                  split="test")
            logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

            results['test_avg_loss'].append(avg_loss)
            results['test_avg_acc'].append(avg_acc)

            _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, args.num_client, hnet, net, criteria, device,
                                                         split="val")
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
            name_save = args.save_path + name_add + '.csv'
            my_csv.to_csv(name_save, index=False)
            with open(args.save_path + name_add + '.pt', 'wb') as f:  # bd0.5_cr0_double bd0.1_cr2
                torch.save([hnet, net], f)

        print('Finish one step in ', time.time() - start_time)


if __name__ == '__main__':
    args = parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    if args.dataset == 'cifar10':
        args.classes_per_node = 2
    else:
        args.classes_per_node = 10
    print("Class per nodes are:", args.classes_per_node)
    train(args, device)
