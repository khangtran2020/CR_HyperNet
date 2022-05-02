import numpy as np

import torch

from Data.node_noniid import *
from Models.models import *
# from MomentAccountant.get_priv import *
from Robustness.robustness import *
from Utils.utils import *
from config import parse_args

def run(args, device):

    # Init data
    nodes = BaseNodes(args.dataset, args.data_path, args.num_client, classes_per_node=args.classes_per_node,
                      batch_size=args.batch_size)

    if args.embed_dim == -1:
        logging.info("auto embedding size")
        args.embed_dim = int(1 + args.num_client / 4)

    if args.dataset == "cifar10":
        hnet = CNNHyper(args.num_client, args.embed_dim, hidden_dim=args.hyper_hid, n_hidden=args.n_hidden,
                        n_kernels=args.nkernels)
        net = CNNTarget(n_kernels=args.nkernels)
    elif args.dataset == "cifar100":
        hnet = CNNHyper(args.num_client, args.embed_dim, hidden_dim=args.hyper_hid,
                        n_hidden=args.n_hidden, n_kernels=args.nkernels, out_dim=100)
        net = CNNTarget(n_kernels=args.nkernels, out_dim=100)
    else:
        raise ValueError("choose dataset from ['cifar10', 'cifar100']")

    new_set_params = {}
    for key in hnet.state_dict():
        value = hnet.state_dict()[key]
        new_set_params[key] = torch.cat(3*[value])
        print(value.size(), new_set_params[key].size())
    exit()

    if args.mode == 'train':
        if args.train_mode == 'clean':
            train_clean(args=args,device=device,nodes=nodes,hnet=hnet,net=net)
        elif args.train_mode == 'userdp':
            train_userdp(args=args, device=device, nodes=nodes, hnet=hnet, net=net)


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
    run(args, device)