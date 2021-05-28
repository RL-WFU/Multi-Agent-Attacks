import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import os
import matplotlib.pyplot as plt
import maddpg_implementation.maddpg.common.tf_util as U
from maddpg_implementation.maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import joblib


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out
"""
def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env
"""

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.all_act_space(), i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.all_act_space(), i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

"""
def test(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir

        print('Loading previous state...')
        #MAKE SURE LOAD_DIR IS WHERE WEIGHTS ARE
        U.load_state(arglist.load_dir+ "policy")

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info

        t_collisions = []
        collisions = []
        min_dist = []
        obs_covered = []

        final_collisions = []
        final_dist = []
        final_obs_cov = []


        transition = []


        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        paths = []
        path_dict = []
        running_paths = [None]

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step



            a_n = []

            for i in range(len(trainers)):
                a_n.append(np.random.choice(np.arange(len(action_n[0])), p=action_n[i]))



            #new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            new_obs_n, rew_n, done_n, info_n = env.step(a_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience

            o = np.asarray(obs_n)
            o_next = np.asarray(new_obs_n)
            o = np.reshape(o, [1, 54])
            o_next = np.reshape(o_next, [1, 54])

            transition.append((o, a_n[0], a_n[1], a_n[2], o_next))

            o1 = np.asarray(obs_n[0])
            o1 = np.reshape(o1, [18,])

            a1 = np.asarray([a_n[0]])

            rew1 = np.asarray([rew_n[0]])

            info1 = np.asarray([info_n['n'][0]])


            if running_paths[0] is None:
                running_paths[0] = dict(
                    observations=[],
                    actions=[],
                    rewards=[],
                    env_infos=[],
                    agent_infos=[],
                    returns=[],
                )

            running_paths[0]["observations"].append(o1)
            running_paths[0]["actions"].append(a1)
            running_paths[0]["rewards"].append(rew1)
            running_paths[0]["env_infos"].append(info1)
            running_paths[0]["agent_infos"].append(info1)
            running_paths[0]["returns"].append(0) #THIS IS FILLER. VALUE SHOULD NOT MATTER







            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:

                paths.append(dict(observations=running_paths[0]["observations"],
                                  actions=running_paths[0]["actions"],
                                  rewards=running_paths[0]["rewards"],
                                  env_infos=running_paths[0]["env_infos"],
                                  agent_infos=running_paths[0]["agent_infos"],
                                  returns=running_paths[0]["returns"],
                                  ))

                running_paths[0] = None

                if len(paths) % 10 == 0 and len(paths) > 1:
                    path_dict.append(dict(paths=paths[-10:]))
                    joblib.dump(path_dict[-1], 'coop_nav/itr_' + str(len(path_dict)-1) + '.pkl')


                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            # COMMENT OUT FOR NON-MADDPG ENVS
            if arglist.benchmark:
                collisions.append(max([info_n['n'][0][1], info_n['n'][1][1], info_n['n'][2][1]]) - 1)

                if train_step > arglist.benchmark_iters and (done or terminal):
                    os.makedirs(os.path.dirname(arglist.benchmark_dir), exist_ok=True)
                    min_dist.append(min([info_n['n'][0][2], info_n['n'][1][2], info_n['n'][1][2]]))
                    obs_covered.append(info_n['n'][0][3])
                    t_collisions.append(sum(collisions))
                    collisions = []



            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):


                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

                final_collisions.append(np.mean(t_collisions[-arglist.save_rate:]))
                final_dist.append(np.mean(min_dist[-arglist.save_rate:]))
                final_obs_cov.append(np.mean(obs_covered[-arglist.save_rate:]))



                os.makedirs(os.path.dirname(arglist.plots_dir), exist_ok=True)
                plt.plot(final_ep_rewards)
                plt.savefig(arglist.plots_dir + arglist.exp_name + '_rewards.png')
                plt.clf()

                plt.plot(final_dist)
                plt.savefig(arglist.plots_dir + arglist.exp_name + '_min_dist.png')
                plt.clf()

                plt.plot(final_obs_cov)
                plt.savefig(arglist.plots_dir + arglist.exp_name + '_obstacles_covered.png')
                plt.clf()

                plt.plot(final_collisions)
                plt.savefig(arglist.plots_dir + arglist.exp_name + '_total_collisions.png')
                plt.clf()

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'

                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)



                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                print()
                print("Average min dist: {}".format(np.mean(final_dist)))
                print("Average number of collisions: {}".format(np.mean(final_collisions)))
                break

        print("Saving Transition...")
        transition = np.asarray(transition)
        print(transition.shape)
        np.save('Transition_new', transition)
        print(transition[-1])

def maddpg_test(arglist):

    test(arglist)
"""

