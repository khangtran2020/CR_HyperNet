import argparse
from Utils.utils import str2bool

def add_general_group(group):
    group.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    group.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    group.add_argument("--save-path", type=str, default="results/", help="dir path for output file")
    group.add_argument("--seed", type=int, default=42, help="seed value")
    group.add_argument("--mode", type=str, default='train', help="Mode of running")
    group.add_argument("--train_mode", type=str, default='clean', help="Mode of training [clean, attack, robust]")
    group.add_argument("--val_mode", type=str, default='clean', help="Mode of validating [clean, robust]")
    group.add_argument("--robustness", type=str, default='false', help="with or withour robustness inference")


def add_data_group(group):
    group.add_argument('--dataset', type=str, default='cifar10', help="used dataset")
    group.add_argument('--num_client', type=int, default=100, help="Number of client in the FL setting")
    group.add_argument('--num_compromised_client', type=int, default=2, help="Number of compromised client in the FL setting")
    group.add_argument('--data_path', type=str, default='data', help="the directory used to save dataset")
    group.add_argument('--data_verbose', action='store_true', help="print detailed dataset info")
    group.add_argument('--save_data', action='store_true')


def add_model_group(group):
    group.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    group.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    group.add_argument("--lr", type=float, default=0.01, help="learning rate")
    group.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    group.add_argument("--inner-wd", type=float, default=5e-5, help="inner weight decay")
    group.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    group.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    group.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    group.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    group.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")
    group.add_argument('--batch_size', type=int, default=100)
    group.add_argument('--dropout', type=float, default=0.5)
    group.add_argument('--train_verbose', action='store_true', help="print training details")
    group.add_argument('--log_every', type=int, default=1, help='print every x epoch')
    group.add_argument('--eval_every', type=int, default=5, help='evaluate every x epoch')
    group.add_argument('--clean_model_save_path', type=str, default='../save/model/clean')
    group.add_argument('--save_clean_model', action='store_true')
    group.add_argument("--bt", type=int, default=10, help="batch")
    group.add_argument("--num_steps", type=int, default=25000)
    group.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    group.add_argument("--inner-steps", type=int, default=4, help="number of inner steps")



def add_atk_group(group):
    pass


def add_defense_group(group):
    group.add_argument('--grad_clip', type=float, default=0.1, help="clipping bound for user-dp")
    group.add_argument('--noise_scale', type=float, default=0.1, help="noise scale for user-dp")
    group.add_argument('--num_comp_cli', type=int, default=1, help="number of compromised client for user-dp")
    group.add_argument('--udp_delta', type=float, default=0.0001, help="broken probability of userdp")
    group.add_argument('--udp_epsilon', type=float, default=1, help="privacy budget of userdp")
    group.add_argument('--attack_norm_bound', type=float, default=1, help="bound of number of compromised client")
    group.add_argument('--num_draws_udp', type=float, default=1000, help="number of draws for userdp robustness bound")
    group.add_argument('--num_draws_ldp', type=float, default=1000, help="number of draws for LDP robustness bound")
    group.add_argument('--eps_userdp', type=float, default=1.0, help="privacy budget for user DP")
    group.add_argument('--eps_ldp', type=float, default=1.0, help="privacy budget for LDP")
    group.add_argument('--model_file', type=str, default=None, help="File to the model")
    group.add_argument('--robustness_confidence_proba', type=float, default=0.05, help="confidence intervals")


def parse_args():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")
    atk_group = parser.add_argument_group(title="Attack-related configuration")
    general_group = parser.add_argument_group(title="General configuration")
    defense_group = parser.add_argument_group(title="Defense configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_atk_group(atk_group)
    add_general_group(general_group)
    add_defense_group(defense_group)
    return parser.parse_args()
