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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def parse_maddpg_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100, help="number of episodes")
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
    parser.add_argument("--save-rate", type=int, default=10,
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


def timed_attack(arglist, thresh, random, rate):
    env = make_env('simple_spread')

    marl_sess = U.single_threaded_session()

    attack_threshold = thresh

    attack_type = random #0 for real policy, 1 for predicted policy, 2 for AIRL policy

    with marl_sess as sess:

        model = build_model("Prediction")
        U.initialize()
        model.load_weights("Prediction_weights/actionMultiClassNetwork")

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
        U.load_state(arglist.load_dir + "policy", saver=trainer_saver)

        env = TfEnv(env)

        # policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
        # U.initialize()

        # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy")
        # saver = tf.train.Saver(vars)
        # U.load_state("data/coop_nav_gcl/weights/policy-150", saver=saver)
        # print("Weights restored")

        episode_rewards = [0.0]
        iter_rewards = []
        train_step = 0
        episode_step = 0

        obs = env.reset()

        transition = []

        attack_percentage = []
        num_attacks = 0

        while True:


            probs = [agent.action(o) for (o, agent) in zip(obs, trainers)]

            actions = []

            for j in range(len(trainers)):
                actions.append(np.random.choice(np.arange(len(probs[0])), p=probs[j]))

            # agent_probs = np.squeeze(policy.get_actions(np.reshape(obs[0], newshape=[1, len(obs[0])]))[1]['prob'])
            # actions[0] = np.random.choice(np.arange(len(agent_probs)), p=agent_probs)

            if not attack_type:
                max_prob = np.max(probs[0])
                min_prob = np.min(probs[0])

                c = max_prob - min_prob

                if c > attack_threshold:
                    actions[0] = np.argmin(probs[0])
                    num_attacks += 1
            else:
                if np.random.random() < rate:
                    actions[0] = np.argmin(probs[0])
                    num_attacks += 1



            new_obs, rew, done, info = env.step(actions)

            # o1 = np.reshape(np.concatenate((obs, [[actions[0]]])))
            # new_obs = np.reshape(np.asarray(new_obs), [1, 54])

            episode_rewards[-1] += rew[0]

            episode_step += 1
            train_step += 1
            done = all(done)
            terminal = (episode_step >= arglist.max_episode_len)

            o = np.asarray(obs)
            o_next = np.asarray(new_obs)
            o = np.reshape(o, [1, 54])
            o_next = np.reshape(o_next, [1, 54])

            transition.append((o, actions[0], actions[1], actions[2], o_next))

            obs = new_obs

            if done or terminal:
                attack_percentage.append(num_attacks/25)
                num_attacks = 0
                obs = env.reset()
                episode_step = 0
                episode_rewards.append(0)

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

        print("AVERAGE REWARD: {}".format(sum(episode_rewards) / len(episode_rewards)))
        print("ATTACK PERCENTAGE: {}".format(sum(attack_percentage) / len(attack_percentage)))

        # print("Saving Transition...")
        # transition = np.asarray(transition)
        # print(transition.shape)
        # np.save('Transition_Attack_Policy', transition)

if __name__ == "__main__":
    arglist = parse_maddpg_args()
    timed_attack(arglist)