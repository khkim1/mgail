#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import gym
import numpy as np
import tensorflow as tf
from algo.ppo import PPOTrain
from network_models.policy_net import Model_net, Policy_net
from time import sleep


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', default=0, type=int)
    parser.add_argument('--train', default=0, type=int)
    parser.add_argument('--logdir', help='log directory', default='log/train/model')
    parser.add_argument('--res_dir', help='filename of model to test', default='trained_models/ppo/expert/model.ckpt')
    parser.add_argument('--use_exp', help='use expert policy', default=1, type=int)
    parser.add_argument('--use_lm', help='use learned model', default=1, type=int)
    parser.add_argument('--data_type', help='use data from random, mediocre, expert', default='expert', type=str)
    parser.add_argument('--play_data', help='play training data', default=0, type=int)
    parser.add_argument('--epoch', help='num epochs of training', default=100, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--rs', help='render speed', default=0.1, type=float)
    parser.add_argument('--depth', help='model depth', default=2, type=int)
    parser.add_argument('--num_hidden', help='number of hidden units', default=64, type=int)

    return parser.parse_args()

# Parse cmdline arg
args = argparser()
render = args.render
train = args.train
data_type = args.data_type
play_data = args.play_data

# Gather expert trajectories
if data_type == 'random':
    exp_o = np.genfromtxt('rtrajectory/observations.csv')
    exp_a = np.expand_dims(np.genfromtxt('rtrajectory/actions.csv', dtype=np.int32), axis=1)
    reset_idx = np.genfromtxt('rtrajectory/reset_idx.csv', dtype=np.float32)
elif data_type == 'med':
    exp_o = np.genfromtxt('mtrajectory/observations.csv')
    exp_a = np.expand_dims(np.genfromtxt('mtrajectory/actions.csv', dtype=np.int32), axis=1)
    reset_idx = np.genfromtxt('mtrajectory/reset_idx.csv', dtype=np.float32)
elif data_type == 'expert':
    exp_o = np.genfromtxt('trajectory/observations.csv')
    exp_a = np.expand_dims(np.genfromtxt('trajectory/actions.csv', dtype=np.int32), axis=1)
    reset_idx = np.genfromtxt('trajectory/reset_idx.csv', dtype=np.float32)
else:
    print("[train_model.py] Error: Unrecognized data type: {}".format(data_type))
assert len(exp_o) == len(exp_a)

# Train a model from expert demonstrations
num_obs_per_state = 1
num_demo = len(exp_o) - num_obs_per_state
print("observation data size: {}".format(exp_o.shape))
print("action data size: {}".format(exp_a.shape))
sleep(1)
given = np.concatenate([exp_o[i:-num_obs_per_state+i] for i in range(num_obs_per_state)] + \
                       [exp_a[i:-num_obs_per_state+i] for i in range(num_obs_per_state)], axis=1)

# Process masked true values
tv = exp_o[num_obs_per_state:]
mask = reset_idx[num_obs_per_state:]
#mask = np.array([i % 200 != 0 for i in range(len(exp_o))], dtype=np.float32)[num_obs_per_state:]
#comp_mask = reset_idx[num_obs_per_state:]
#print("Array comparison: {}".format(np.array_equal(mask, comp_mask)))
tv = np.concatenate([tv, mask[:, np.newaxis]], axis=1)

assert num_demo == len(tv) == len(given)

# Train loop parameters
num_epochs = args.epoch
batch = 64
batches_per_epoch = 100
lr = args.lr
rs = args.rs
use_lm = args.use_lm
depth = args.depth
num_hidden = args.num_hidden

# Initialize the environment
env = gym.make('CartPole-v0')
name = 'model'
model = Model_net(name, env, num_obs_per_state, lr, depth, num_hidden)

# Load expert policy
Policy = Policy_net('policy', env)
restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
saver = tf.train.Saver(var_list=restore_vars)

# Play the data
if play_data:
    for i in range(len(exp_o)):
        env.render()
        env.env.state = exp_o[i]

with tf.Session() as sess, tf.summary.FileWriter(args.logdir, sess.graph) as writer:
    sess.run(tf.global_variables_initializer())
    if train:
        for epoch in range(num_epochs):
            epoch_loss = []
            for i in range(batches_per_epoch):
                batch_idx = np.random.randint(num_demo, size=batch)
                batch_given = given[batch_idx]
                batch_tv = tv[batch_idx]
                _, loss = model.train_sl(batch_given, batch_tv)
                epoch_loss.append(loss)

            print("Epoch: {}, Loss: {}".format(epoch, np.mean(epoch_loss)))
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='epoch_loss', simple_value=np.mean(epoch_loss))]), epoch)

    # Render loop
    if render:
        saver.restore(sess, args.res_dir)

        while True:
            obs = [env.reset()]
            a_0 = np.array([0])
            action = [a_0]

            '''
            a_0 = np.array([0])
            a_1 = np.array([1])
            action = [a_0, a_1, a_0, a_1] # Starting action sequence

            for i in range(num_obs_per_state-1):
                cur_obs, _, _, _ = env.step(action[i][0])
                obs.append(cur_obs)
            '''

            for i in range(500):
                # Render
                env.render()
                sleep(rs)

                given = np.expand_dims(np.concatenate(obs+action, axis=0), axis=0)

                if use_lm:
                    pred_obs = np.squeeze(model.step(given))
                    env.env.state = pred_obs
                else:
                    pred_obs, _, _, _ = env.step(action[-1][0])

                obs.pop(0)
                action.pop(0)
                obs.append(pred_obs)

                # Process the actions
                if args.use_exp:
                    new_a, _ = Policy.act(obs=np.expand_dims(pred_obs, axis=0), stochastic=True)
                else:
                    new_a = [env.action_space.sample()]

                action.append(np.array(new_a))

                #print("i: {}, Action: {}, Obs: {}".format(i, new_a, obs[-1]))

            '''
            #_, _, done, _ = env.step(env.action_space.sample())
            print(env.env.state)
            env.env.state = exp_o[i]
            #env.render()
            #sleep(1)
            '''

