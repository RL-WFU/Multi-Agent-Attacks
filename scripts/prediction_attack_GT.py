from Predictiton_based_attacks.train_prediction_net import build_model
import tensorflow as tf
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.envs.multiagent.make_coop_env import *
import maddpg_implementation.maddpg.common.tf_util as U
from maddpg_implementation.experiments.test import get_trainers
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from sandbox.rocky.tf.envs.base import TfEnv
import time
import copy

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def parse_maddpg_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_adversary", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="coop_nav", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./weights_new/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="../maddpg_implementation/Good_agent_weights/",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=-1,
                        help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./data/oop_nav_attack/",
                        help="directory where plot data is saved")

    parser.add_argument("--att-benchmark-dir", type=str, default="Attack/benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--att-plots-dir", type=str, default="Attack/learning_curves/",
                        help="directory where plot data is saved")

    return parser.parse_args()


def get_logits(model, obses, actions, in_length):
    obs = np.reshape(obses, [1, 3, in_length])
    action = np.reshape(actions, [1, 3, 1])
    model_input = np.concatenate([obs, action], axis=2)
    logits = np.asarray(model.predict_on_batch(model_input.astype('float64')))
    return logits


def attack(arglist, threshold=0, attack_rate=1, test=False):
    scenario = arglist.scenario
    env, og_scenario, og_world = make_env(scenario, return_ws=True, benchmark=True)

    prediction_envs = []
    prediction_worlds = []
    prediction_scenarios = []
    for i in range(5):
        e, s, w = make_env(scenario, return_ws=True, benchmark=True)
        prediction_envs.append(TfEnv(e))
        prediction_scenarios.append(s)
        prediction_worlds.append(w)

    tf.reset_default_graph()
    marl_sess = U.single_threaded_session()

    attacking = True
    attack_threshold = threshold

    with marl_sess as sess:

        obs_shape_n = [env.all_obs_space[i].shape for i in range(env.n)]
        num_adversaries = 0
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        U.initialize()
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir

        trainer_vars = []
        for i in range(env.n):
            trainer_vars.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_%d" % i))

        trainer_vars = np.asarray(trainer_vars)
        trainer_vars = trainer_vars.flatten()
        trainer_vars = list(trainer_vars)
        trainer_saver = tf.train.Saver(trainer_vars)

        if scenario == 'simple_adversary':
            U.load_state(arglist.load_dir + "adv_policy", saver=trainer_saver)
        elif scenario == 'simple_spread':
            U.load_state(arglist.load_dir + "nav_policy", saver=trainer_saver)

        env = TfEnv(env)

        controlled_agent = 0
        if scenario == "simple_adversary":
            controlled_agent = 1

        episode_rewards = [0.0]
        iter_rewards = []
        train_step = 0
        episode_step = 0

        obs = env.reset()

        episode_obs = []
        episode_acts = []

        transition = []

        episode_attacks = 0
        avg_episode_attacks = []

        adv_rewards = [0.0]
        coop_rewards = [0.0]

        collisions = 0
        min_dists = 0
        occ_landmarks = 0
        avg_collisions = []
        avg_min_dists = []
        avg_occ_landmarks = []

        adv_dist = 0
        coop_dist = 0
        avg_adv_dist = []
        avg_coop_dist = []
        label = 0

        while True:
            label = 0

            episode_obs.append(obs)

            probs = [agent.action(o) for (o, agent) in zip(obs, trainers)]

            actions = []

            for j in range(len(trainers)):
                actions.append(np.random.choice(np.arange(len(probs[0])), p=probs[j]))

            a_n = []
            all_next_obs = []
            for i in range(len(trainers)):
                a_n.append(actions[i])

            copy_worlds = [copy.deepcopy(og_world) for i in range(5)]
            for i in range(5):
                prediction_scenarios[i].copy_world(prediction_worlds[i], copy_worlds[i])

                a_n[controlled_agent] = i
                next_obs, _, _, _ = prediction_envs[i].step(a_n)

                all_next_obs.append(next_obs)

            #Shape (5, 3, 5). First axis is next_o according to action index.
            #Second axis is agent. Third axis is probability according to action index
            next_probs = [[agent.action(n_o) for (n_o, agent) in zip(all_next_obs[j], trainers)] for j in range(len(probs[0]))]
            next_probs = np.asarray(next_probs)

            #Probabilities for other agents given agent 0 action
            #Shape (5, 2, 5). Axis 0 is the action index. Axis 1 is the good agent index. Axis 2 is the action index for probs
            if scenario == "simple_spread":
                correct_next_probs = next_probs[:, 1:, :]
            else:
                correct_next_probs = np.concatenate([next_probs[:, 0, :], next_probs[:, 2, :]], axis=1)
                correct_next_probs = np.reshape(correct_next_probs, [5, 2, 5])


            kl_divergences = []
            #Loop through each bad action
            for i in range(5):
                #Compute KL divergence of probs resulting for bad actions for both good agents
                total_kl = 0
                adv_kl = 0
                coop_kl = 0
                for j in range(2):
                    agent_kl = 0
                    for k in range(5):
                        log_term = np.log(correct_next_probs[actions[controlled_agent], j, k] / correct_next_probs[i, j, k])
                        agent_kl += correct_next_probs[actions[controlled_agent], j, k] * log_term
                    if j == 0:
                        adv_kl += agent_kl
                    else:
                        coop_kl += agent_kl

                    total_kl += agent_kl

                if scenario == 'simple_adversary':
                    kl_divergences.append(coop_kl)
                else:
                    kl_divergences.append(total_kl)

            #print(kl_divergences)

            """
            #KL_divergences will contain 5 kl divergences. They each have a divergence measure which
            between the correct next_probs (given a good agent0 action) and the next_probs retrieved
            by agent0 taking action i at the current timestep, where i is the action index
            """

            worst_action = np.argmax(kl_divergences)

            if attacking and kl_divergences[int(worst_action)] > attack_threshold:
                actions[controlled_agent] = worst_action
                episode_attacks += 1
                label = 1









            episode_acts.append(actions[controlled_agent])  # for prediction network

            new_obs, rew, done, info_n = env.step(actions)

            o = np.concatenate(obs).ravel()
            o_next = np.concatenate(new_obs).ravel()
            o = np.reshape(o, [1, -1])
            o_next = np.reshape(o_next, [1, -1])

            # transition.append((o, a_n[controlled_agent], o_next, label))
            transition.append((o, actions[0], actions[1], actions[2], o_next, label))

            # o1 = np.reshape(np.concatenate((obs, [[actions[0]]])))
            # new_obs = np.reshape(np.asarray(new_obs), [1, 54])

            if arglist.scenario == 'simple_spread':
                collisions += max([info_n['n'][0][1], info_n['n'][1][1], info_n['n'][2][1]]) - 1
                min_dists += info_n['n'][0][2]
                occ_landmarks += info_n['n'][0][3]

            else:
                adv_dist += info_n['n'][0]
                coop_dist += min(info_n['n'][1][2], info_n['n'][2][2])

            if scenario == 'simple_adversary':
                for i, r in enumerate(rew):
                    if i == 0:
                        adv_rewards[-1] += r
                    else:
                        coop_rewards[-1] += r
                    episode_rewards[-1] += r
            else:
                episode_rewards[-1] += rew[0]


            episode_step += 1
            train_step += 1
            done = all(done)
            terminal = (episode_step >= arglist.max_episode_len)

            """
            o = np.asarray(obs)
            o_next = np.asarray(new_obs)
            o = np.reshape(o, [1, 54])
            o_next = np.reshape(o_next, [1, 54])

            transition.append((o, actions[0], actions[1], actions[2], o_next))
            """

            obs = new_obs

            if done or terminal:
                avg_episode_attacks.append(episode_attacks / 25)
                episode_obs = []
                episode_acts = []

                label = 0

                avg_collisions.append(collisions / arglist.max_episode_len)
                avg_min_dists.append(min_dists / arglist.max_episode_len)
                avg_occ_landmarks.append(occ_landmarks / arglist.max_episode_len)

                avg_adv_dist.append(adv_dist / arglist.max_episode_len)
                avg_coop_dist.append(coop_dist / arglist.max_episode_len)

                collisions = 0
                min_dists = 0
                occ_landmarks = 0

                adv_dist = 0
                coop_dist = 0

                obs = env.reset()
                episode_step = 0
                episode_attacks = 0
                episode_rewards.append(0)
                adv_rewards.append(0)
                coop_rewards.append(0)

            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            if terminal and len(episode_rewards) % arglist.save_rate == 0:
                print("steps: {}, episodes: {}, mean episode reward: {}".format(
                    train_step, len(episode_rewards),
                    sum(episode_rewards[-arglist.save_rate - 1:-1]) / len(episode_rewards[-arglist.save_rate - 1:-1]))
                )
                iter_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))

                """
                os.makedirs(os.path.dirname(arglist.plots_dir), exist_ok=True)

                plt.plot(iter_rewards)

                if attacking:
                    plt.savefig(arglist.plots_dir + arglist.exp_name + "_ATTACK_REWARDS.png")
                else:
                    plt.savefig(arglist.plots_dir + arglist.exp_name + "_COOP_REWARDS.png")
                """
            if len(episode_rewards) > arglist.num_episodes:
                break

        if arglist.scenario == 'simple_adversary':
            good_agent_rewards = (sum(coop_rewards) / len(coop_rewards))
            print("AVERAGE GOOD AGENT REWARD: {}".format(good_agent_rewards))
            print("AVERAGE ADV AGENT REWARD: {}".format(sum(adv_rewards) / len(adv_rewards)))
        else:
            print("AVERAGE REWARD: {}".format(sum(episode_rewards) / len(episode_rewards)))
        print("ATTACK RATE: {}".format(sum(avg_episode_attacks) / len(avg_episode_attacks)))

        if arglist.scenario == 'simple_spread':
            print("Average collisions: {}".format(sum(avg_collisions) / len(avg_collisions)))
            print("Average total min dist to targets: {}".format(sum(avg_min_dists) / len(avg_min_dists)))
            print("Average Occupied Landmarks: {}".format(sum(avg_occ_landmarks) / len(avg_occ_landmarks)))
        else:
            print("Average Adv Dist to Target: {}".format(sum(avg_adv_dist) / len(avg_adv_dist)))
            print("Average Best Coop Dist to Target: {}".format(sum(avg_coop_dist) / len(avg_coop_dist)))



        print("Saving Transition...")
        transition = np.asarray(transition)
        print(transition.shape)
        if test:
            np.save('phys_decept_whitebox_prediction_{}_test'.format(int(attack_rate * 100)), transition)
        else:
            #np.save('phys_decept_whitebox_prediction_{}'.format(int(attack_rate * 100)), transition)
            pass
        # print("Saving Transition...")
        # transition = np.asarray(transition)
        # print(transition.shape)
        # np.save('Transition_Attack_Policy', transition)

        print("Transition positive rate: {}".format(np.sum(transition[:, 5]) / (25 * arglist.num_episodes)))

        """
        PREDICTION ACCURACY:
        [-0.18251245  0.05932195  0.01846297  0.01988064  0.08484674] Agent 2
        [-0.17727599  0.03394374  0.05455007  0.04836935  0.04041254] Agent 3

        It seems the network struggles to predict when the agents stay still
            - Off by about 20% on average for action 0
        Rest of the errors are probably due to the stay still action being predicted incorrectly
            - Aside from staying still, the results are perfect
        """


if __name__ == "__main__":
    arglist = parse_maddpg_args()
    attack(arglist)