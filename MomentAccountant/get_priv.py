import os
import sys

path = "/".join([x for x in os.path.realpath(__file__).split('/')[:-2]])
sys.path.insert(0, path)
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from gaussian_moments import *
import time
import accountant, utils
import pandas as pd
from config import parse_args


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);

def idle():
    return


# compute sigma using strong composition theory given epsilon
def compute_sigma(epsilon, delta):
    return 1 / epsilon * np.sqrt(np.log(2 / math.pi / np.square(delta)) + 2 * epsilon)


# compute sigma using moment accountant given epsilon
def comp_sigma(q, T, delta, epsilon):
    c_2 = 4 * 1.26 / (0.01 * np.sqrt(10000 * np.log(100000)))  # c_2 = 1.485
    return c_2 * q * np.sqrt(T * np.log(1 / delta)) / epsilon


# compute epsilon using abadi's code given sigma
def comp_eps(lmbda, q, sigma, T, delta):
    lmbds = range(1, lmbda + 1)
    log_moments = []
    for lmbd in lmbds:
        log_moment = compute_log_moment(q, sigma, T, lmbd)
        log_moments.append((lmbd, log_moment))

    eps, delta = get_privacy_spent(log_moments, target_delta=delta)
    return eps


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1);
    return tf.Variable(initial);


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape);
    return tf.Variable(initial);


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME');


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME');


FLAGS = None;

# target_eps = [0.125,0.25,0.5,1,2,4,8]
# target_eps =[0.5, 1, 2, 4, 6, 8]  #[0.5, 1, 2, 3, 4];
target_eps = [1.34, 1.35, 1.36]  # [0.5, 1, 2, 3, 4];


def main(args):
    small_num = 1e-5  # 'a small number'
    large_num = 1e5  # a large number'

    z = args.noise_scale
    sigma = z  # 'noise scale'

    delta = args.udp_delta  # 'delta'

    clip = 1  # 'whether to clip the gradient'

    D = args.num_client
    batch_size = args.bt
    sample_rate = batch_size / D  # 'sample rate q = L / N'
    num_steps = args.num_steps  # 'number of steps T = E * N / L = E / q'
    result_path = 'check_priv_spent_nscale' + str(sigma) + '_D' + str(D) + '_bs' + str(batch_size)  # + '.txt'

    '''from tensorflow.examples.tutorials.mnist import input_data;
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True);'''

    for target in target_eps:
        print('target ', target)
        sess = tf.InteractiveSession()

        priv_accountant = accountant.GaussianMomentsAccountant(D)
        privacy_accum_op = priv_accountant.accumulate_privacy_spending([None, None], sigma, batch_size)

        # sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        # with open(result_path, 'a') as outf:
        #     outf.write('delta target {} \n'.format(delta))

        iter_ = []
        eps_ = []
        delta_ = []
        for i in range(num_steps):  # range(num_steps):
            sess.run([privacy_accum_op])
            spent_eps_deltas = priv_accountant.get_privacy_spent(sess, target_eps=[target])
            print(i, spent_eps_deltas)
            # exit()
            # with open(result_path, 'a') as outf:
            #     outf.write('| step {} | priv {}\n'.format(i, spent_eps_deltas))
            #     # outf.write('=' * 8)
            iter_.append(i)
            eps_.append(spent_eps_deltas[0][0])
            delta_.append(spent_eps_deltas[0][1])

            _break = False
            for _eps, _delta in spent_eps_deltas:
                if _delta >= delta:
                    _break = True
                    break
            if _break == True:
                break
        data_w = {'epoch': iter_, 'eps': eps_, 'delta spent': delta_}
        my_csv = pd.DataFrame(data_w)
        my_csv.to_csv(result_path + '_epoch' + str(i) + '_eps' + str(target) + '_delta1e-5.csv', index=False)
        ###


args = parse_args()
main(args)
# if __name__ == '__main__':
#   tf.app.run()
