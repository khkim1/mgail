#!/usr/bin/python3
import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Model_Policy_net, Policy_net, Model_net
from network_models.discriminator import Discriminator, ModelDiscriminator
from algo.ppo import PPOTrain, ModelTrain
import pdb
import math
from time import sleep


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail_model3')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail_model')
    parser.add_argument('--gamma', default=0)
    parser.add_argument('--iteration', default=int(1e6), type=int)
    parser.add_argument('--resdir', help='expert actor policy', default='trained_models/ppo/expert/model.ckpt')
    return parser.parse_args()


def check_done(state, policy_steps):
    x, x_dot, theta, theta_dot = state
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4

    done = x < -x_threshold \
           or x > x_threshold \
           or theta < -theta_threshold_radians \
           or theta > theta_threshold_radians

    if policy_steps > 200:
        done = True

    return done

def check_done_easy(policy_steps):
    return policy_steps > 200

if __name__ == '__main__':
    args = argparser()
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space

    obs_dim = 4
    act_dim = 1
    num_epochs = 1
    batches_per_epoch = 10
    batch = 64
    rl = True
    sl = False
    render = True
    render_freq = 500
    d_freq = 1
    use_dummy_data = False
    use_random = False

    # For expert data it seems that gamma = 0, (rl: (32, 5)), stop=50 works well
    stochastic_policy = True
    stochastic_model = True

    disc_test = False

    # Policy is now the dynamics model
    Model = Model_Policy_net('model_policy', env, obs_dim+act_dim, obs_dim)
    Old_Model = Model_Policy_net('old_model_policy', env, obs_dim+act_dim, obs_dim)
    PPO = ModelTrain(Model, Old_Model, obs_dim+act_dim, obs_dim, gamma=args.gamma)
    D = ModelDiscriminator(env, obs_dim+act_dim, obs_dim)

    # Load the actor
    Policy = Policy_net('policy', env)
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
    restorer = tf.train.Saver(var_list=restore_vars)
    saver = tf.train.Saver()

    # Process expert data for discriminator
    if use_dummy_data:
        # Dummy expert data (REMOVE WHEN DONE DEBUGGING)
        print("Using dummy data")
        load_exp_o = np.tile(np.array([[0., 0., 1., 0.]]), [10000, 1])
        load_exp_a = np.ones((10000, act_dim))
        exp_o = np.concatenate([load_exp_o, load_exp_a], axis=1)
        exp_a = load_exp_o
        reset_idx = np.zeros((len(exp_o)))

    elif use_random:
        print("Using random policy data")
        load_exp_o = np.genfromtxt('rtrajectory/observations.csv')
        load_exp_a = np.expand_dims(np.genfromtxt('rtrajectory/actions.csv', dtype=np.int32), axis=1)
        reset_idx = np.genfromtxt('rtrajectory/reset_idx.csv', dtype=np.float32)
        exp_o = np.concatenate([load_exp_o[:-1]] + [load_exp_a[:-1]]*act_dim, axis=1)
        exp_a = load_exp_o[1:]

        # Take out transitions at the end of episodes
        mask = reset_idx[1:]
        delete_idx = np.nonzero(np.logical_not(mask))[0]
        exp_o = np.delete(exp_o, delete_idx, 0)
        exp_a = np.delete(exp_a, delete_idx, 0)

    else:
        print("Using expert data")
        load_exp_o = np.genfromtxt('trajectory/observations.csv')
        load_exp_a = np.expand_dims(np.genfromtxt('trajectory/actions.csv', dtype=np.int32), axis=1)
        reset_idx = np.genfromtxt('trajectory/reset_idx.csv', dtype=np.float32)
        exp_o = np.concatenate([load_exp_o[:-1]] + [load_exp_a[:-1]]*act_dim, axis=1)
        exp_a = load_exp_o[1:]

        # Take out transitions at the end of episodes
        mask = reset_idx[1:]
        delete_idx = np.nonzero(np.logical_not(mask))[0]
        exp_o = np.delete(exp_o, delete_idx, 0)
        exp_a = np.delete(exp_a, delete_idx, 0)

    # Process expert data for supervised learning
    given = np.concatenate([load_exp_o[:-1], load_exp_a[:-1]], axis=1)
    tv = load_exp_o[1:]
    mask = reset_idx[1:]
    tv = np.concatenate([tv, mask[:, np.newaxis]], axis=1)
    num_demo = len(load_exp_o) - 1
    print("Number of Demonstrations: {}".format(num_demo))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restorer.restore(sess, args.resdir)
        writer = tf.summary.FileWriter(args.logdir, sess.graph)

        success_num = 0

        for iteration in range(args.iteration):
            # Supervised Learning
            if sl:
                epoch_loss = []
                for i in range(batches_per_epoch):
                    batch_idx = np.random.randint(num_demo, size=batch)
                    batch_given = given[batch_idx]
                    batch_tv = tv[batch_idx]
                    _, loss = Model.train_sl(batch_given, batch_tv)
                    epoch_loss.append(loss)

                #print("Epoch: {}, Loss: {}".format(iteration, np.mean(epoch_loss)))
                #sl = False

            if render and iteration % render_freq == 0:
                obs = np.expand_dims(env.reset(), axis=0)
                for i in range(100):
                    # Render
                    env.render()
                    sleep(0.01)

                    # Process the actions
                    if use_dummy_data:
                        action = np.array([1.])
                    elif use_random:
                        #action = np.array([np.random.randint(2)])
                        action = np.array([0.])
                    else:
                        action, _ = Policy.act(obs=obs, stochastic=stochastic_policy)
                        action = np.array(action)

                    # State input to model
                    given_model = np.expand_dims(np.concatenate([np.squeeze(obs)]+[action]*act_dim, axis=0), axis=0)
                    obs, _ = Model.step(given_model, stochastic=True)
                    env.env.state = np.squeeze(obs)

                    print(given_model)

            if rl:
                # Reinforcement Learning
                observations = []
                actions = []
                rewards = []
                v_preds = []
                run_policy_steps = 0
                obs = np.expand_dims(env.reset(), axis=0)

                while True:
                    #env.render()
                    run_policy_steps += 1
                    if use_dummy_data:
                        act = np.array([[1.]])
                    elif use_random:
                        #act = np.array([[np.random.randint(2)]])
                        act = np.array([[0.]])
                    else:
                        act, _ = Policy.act(obs=obs, stochastic=stochastic_policy)
                        act = np.expand_dims(act, axis=0)

                    # Model state
                    state = np.concatenate([obs] + [act]*act_dim, axis=1)
                    assert state.shape[1] == obs_dim + act_dim

                    # Take a step with the model
                    if disc_test:
                        next_obs, reward, done, info = env.step(np.asscalar(act))
                        next_obs = np.expand_dims(next_obs, axis=0)
                        v_pred = 0
                    else:
                        next_obs, v_pred = Model.step(state, stochastic=stochastic_model)
                        v_pred = np.asscalar(v_pred)
                        #done = check_done(np.squeeze(next_obs), run_policy_steps)
                        done = check_done_easy(run_policy_steps)
                    reward = 1. if not done else 0.

                    observations.append(state)
                    actions.append(next_obs)
                    rewards.append(reward)
                    v_preds.append(v_pred)

                    if done:
                        _, v_pred = Policy.act(obs=next_obs, stochastic=stochastic_policy)
                        v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                        break
                    else:
                        obs = next_obs


                # Summary
                el_sum = tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                er_sum = tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                writer.add_summary(el_sum, iteration)
                writer.add_summary(er_sum, iteration)

                '''
                # Finished check
                if sum(rewards) >= 195:
                    success_num += 1
                    if success_num >= 100:
                        saver.save(sess, args.savedir + '/model.ckpt')
                        print('Clear!! Model saved.')
                        break
                else:
                    success_num = 0
                '''

                # convert list to numpy array for feeding tf.placeholder
                observations = np.reshape(observations, newshape=[-1] + [obs_dim + act_dim])
                actions = np.reshape(actions, newshape=[-1] + [obs_dim])

                '''
                for j in range(len(observations)):
                    env.env.state = observations[j, :4]
                    env.render()
                '''

                # output of this discriminator is reward
                d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
                e_rewards = D.get_rewards(agent_s=exp_o, agent_a=exp_a)
                d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)

                '''
                # train discriminator
                if iteration % d_freq == 0 and np.mean(e_rewards) / np.mean(d_rewards) < 1.5:
                    for epoch in range(1):
                        for k in range(1):
                            batch_idx_exp = np.random.randint(num_demo, size=batch)
                            batch_idx = np.random.randint(len(observations), size=batch)
                            #batch_idx = np.random.randint(num_demo, size=batch)

                            batch_exp_o = exp_o[batch_idx_exp]
                            batch_exp_a = exp_a[batch_idx_exp]
                            batch_obs = observations[batch_idx]
                            batch_a = actions[batch_idx]

                            #batch_obs = exp_o[batch_idx]
                            #batch_a = exp_a[batch_idx]

                            D.train(expert_s=batch_exp_o,
                                    expert_a=batch_exp_a,
                                    agent_s=batch_obs,
                                    agent_a=batch_a)
                '''

                if np.mean(e_rewards) / np.mean(d_rewards) < 2.:
                    for k in range(1):
                        D.train(expert_s=exp_o,
                                expert_a=exp_a,
                                agent_s=observations,
                                agent_a=actions)

                # output of this discriminator is reward
                d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
                e_rewards = D.get_rewards(agent_s=exp_o, agent_a=exp_a)
                d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)

                print("Iteration: {}, Rewards: {}, DRewards: {}, ERewards: {}".format(iteration, sum(rewards), np.mean(d_rewards), np.mean(e_rewards)), end='\r')

                if not disc_test:
                    # Advantage estimation
                    gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
                    gaes = np.array(gaes).astype(dtype=np.float32)
                    v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

                    # Train policy
                    inp = [observations, actions, gaes, d_rewards, v_preds_next]
                    PPO.assign_policy_parameters()
                    for epoch in range(5): # 3, 100 (works well): 5, 200 works well
                        sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)  # indices are in [low, high)
                        sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training dat
                        a
                        PPO.train(obs=sampled_inp[0],
                                  actions=sampled_inp[1],
                                  gaes=sampled_inp[2],
                                  rewards=sampled_inp[3],
                                  v_preds_next=sampled_inp[4])

                    summary = PPO.get_summary(obs=inp[0],
                                              actions=inp[1],
                                              gaes=inp[2],
                                              rewards=inp[3],
                                              v_preds_next=inp[4])


                    sl_loss = Model.get_sl_loss(given, tv)

                    writer.add_summary(summary, iteration)
                    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='d_rewards', simple_value=np.mean(d_rewards))]), iteration)
                    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='sl_loss', simple_value=sl_loss)]), iteration)
        writer.close()



