import os
import sys
path = "/".join([x for x in os.path.realpath(__file__).split('/')[:-2]])
sys.path.insert(0, path)
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import logging
import random
from collections import OrderedDict, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import trange
import time
from Data.node_noniid import BaseNodes
from Utils.utils import get_device, set_logger, set_seed, get_gaussian_noise
from copy import deepcopy


class CNNHyper(nn.Module):
    def __init__(
            self, n_nodes, embedding_dim, in_channels=3, out_dim=10, n_kernels=16, hidden_dim=100,
            spec_norm=False, n_hidden=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)
        self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.c2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
        self.l1_weights = nn.Linear(hidden_dim, 120 * 2 * self.n_kernels * 5 * 5)
        self.l1_bias = nn.Linear(hidden_dim, 120)
        self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
        self.l2_bias = nn.Linear(hidden_dim, 84)
        self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 84)
        self.l3_bias = nn.Linear(hidden_dim, self.out_dim)

        if spec_norm:
            self.c1_weights = spectral_norm(self.c1_weights)
            self.c1_bias = spectral_norm(self.c1_bias)
            self.c2_weights = spectral_norm(self.c2_weights)
            self.c2_bias = spectral_norm(self.c2_bias)
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)
            self.l3_weights = spectral_norm(self.l3_weights)
            self.l3_bias = spectral_norm(self.l3_bias)

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(2 * self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(120, 2 * self.n_kernels * 5 * 5),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(84, 120),
            "fc2.bias": self.l2_bias(features).view(-1),
            "fc3.weight": self.l3_weights(features).view(self.out_dim, 84),
            "fc3.bias": self.l3_bias(features).view(-1),
        })

        weights_tensor = torch.cat((self.c1_weights(features).view(-1),
            self.c1_bias(features).view(-1),self.c2_weights(features).view(-1),
            self.c2_bias(features).view(-1),self.l1_weights(features).view(-1),
            self.l1_bias(features).view(-1),self.l2_weights(features).view(-1),
            self.l2_bias(features).view(-1),self.l3_weights(features).view(-1),
            self.l3_bias(features).view(-1)),0)

        return weights, weights_tensor


class CNNTarget(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNNTarget, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class CNNHyper_1D(nn.Module):
    def __init__(
            self, n_nodes, embedding_dim, in_channels=1, out_dim=10, n_kernels=6, hidden_dim=100,
            spec_norm=False, n_hidden=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)
        self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.c2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5*5)
        self.c2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
        self.l1_weights = nn.Linear(hidden_dim, 120 * 2 * self.n_kernels*5*5)
        self.l1_bias = nn.Linear(hidden_dim, 120)
        self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
        self.l2_bias = nn.Linear(hidden_dim, 84)
        self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 84)
        self.l3_bias = nn.Linear(hidden_dim, self.out_dim)

        if spec_norm:
            self.c1_weights = spectral_norm(self.c1_weights)
            self.c1_bias = spectral_norm(self.c1_bias)
            self.c2_weights = spectral_norm(self.c2_weights)
            self.c2_bias = spectral_norm(self.c2_bias)
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)
            self.l3_weights = spectral_norm(self.l3_weights)
            self.l3_bias = spectral_norm(self.l3_bias)

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(2 * self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(120, 2 * self.n_kernels*5*5),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(84, 120),
            "fc2.bias": self.l2_bias(features).view(-1),
            "fc3.weight": self.l3_weights(features).view(self.out_dim, 84),
            "fc3.bias": self.l3_bias(features).view(-1),
        })

        weights_tensor = torch.cat((self.c1_weights(features).view(-1),
            self.c1_bias(features).view(-1),self.c2_weights(features).view(-1),
            self.c2_bias(features).view(-1),self.l1_weights(features).view(-1),
            self.l1_bias(features).view(-1),self.l2_weights(features).view(-1),
            self.l2_bias(features).view(-1),self.l3_weights(features).view(-1),
            self.l3_bias(features).view(-1)),0)

        return weights, weights_tensor


class CNNTarget_1D(nn.Module):
    def __init__(self, in_channels=1, n_kernels=6, out_dim=10):
        super(CNNTarget_1D, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2*n_kernels, 5)
        self.fc1 = nn.Linear(2*n_kernels*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

# training vanilla hypernet
def train_clean(args, device, nodes, hnet, net):
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
            hnet.embeddings.weight.grad = hnet_grads_each[0]
            optimizer.step()

            if c == 0:
                hnet_grads = deepcopy(list(hnet_grads_each))
                for t in range(len(hnet_grads)):
                    hnet_grads[t] = hnet_grads[t] / args.bt
            else:
                tmp = list(hnet_grads_each)
                for t in range(len(hnet_grads)):
                    hnet_grads[t] += tmp[t] / args.bt
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
            step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, args.num_client, hnet, net, criteria, device,
                                                                  split="test")
            logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

            results['test_avg_loss'].append(avg_loss)
            results['test_avg_acc'].append(avg_acc)

            _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, args.num_client, hnet, net, criteria, device, split="val")
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

# training userdp hypernet
def train_userdp(args, device, nodes, hnet, net) -> None:
    hnet = hnet.to(device)
    net = net.to(device)

    ##################
    # init optimizer #
    ##################
    sampling_prob = args.bt/args.num_client
    embed_lr = args.embed_lr if args.embed_lr is not None else args.lr
    optimizers = {
        'sgd': torch.optim.SGD(params=hnet.parameters(), lr=args.lr
            # [
            #     {'params': [p for n, p in hnet.named_parameters() if 'embed' not in n]},
            #     {'params': [p for n, p in hnet.named_parameters() if 'embed' in n], 'lr': embed_lr}
            # ], lr=args.lr  # , momentum=0.9, weight_decay=wd
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

#   General functions for models

