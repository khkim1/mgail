import argparse
import gym
import numpy as np
from network_models.policy_net import Policy_net
import tensorflow as tf
import math


# noinspection PyTypeChecker
def open_file_and_save(file_path, data):
    """
    :param file_path: type==string
    :param data:
    """
    try:
        with open(file_path, 'ab') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')
    except FileNotFoundError:
        with open(file_path, 'wb') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', default=10, type=int)

    return parser.parse_args()


def main(args):
    env = gym.make('CartPole-v0')
    env.env.theta_threshold_radians = 350 * math.pi / 360
    env.seed(0)
    ob_space = env.observation_space
    num_repeat = 4

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        obs = env.reset()

        for iteration in range(args.iteration):  # episode
            observations = []
            actions = []
            reset_idx = []
            run_steps = 0

            while True:
                run_steps += 1
                # prepare to feed placeholder Policy.obs
                obs = np.stack([obs]).astype(dtype=np.float32)

                #act = np.random.randint(2)
                act = 0

                observations.append(obs)
                actions.append(act)

                for i in range(num_repeat):
                    next_obs, reward, done, info = env.step(act)
                    if done:
                        break

                if run_steps == 1:
                    reset_idx += [0]
                else:
                    reset_idx += [1]

                if done:
                    print(run_steps)
                    obs = env.reset()
                    break
                else:
                    obs = next_obs

            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)
            reset_idx = np.array(reset_idx).astype(dtype=np.int32)

            open_file_and_save('rtrajectory/observations.csv', observations)
            open_file_and_save('rtrajectory/actions.csv', actions)
            open_file_and_save('rtrajectory/reset_idx.csv', reset_idx)


if __name__ == '__main__':
    args = argparser()
    main(args)
