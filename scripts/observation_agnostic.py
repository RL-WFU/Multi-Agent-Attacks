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
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model

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


def get_logits(model, obses, in_length, actions=None, model_type="other_policy"):
    if model_type == "other_policy":
        obs = np.reshape(obses, [-1, 3, in_length])
        action = np.reshape(actions, [-1, 3, 1])
        model_input = np.concatenate([obs, action], axis=2)
        logits = np.asarray(model.predict_on_batch(model_input.astype('float64')))
        return logits
    elif model_type == "transition":
        obs = np.reshape(obses, [-1, 3, in_length])
        action = np.reshape(actions, [-1, 3, 3])
        model_input = np.concatenate([obs, action], axis=2)
        logits = np.asarray(model.predict_on_batch(model_input.astype('float64')))
        return logits
    elif model_type == "policy":
        obs = np.reshape(obses, [-1, 3, in_length])
        logits = np.asarray(model.predict_on_batch(obs.astype('float64')))
        return logits
    else:
        raise NotImplementedError


def build_model(scope, in_length, fname=None, model_type="other_policy"):
    if model_type == "other_policy":
        with tf.variable_scope(scope):
            # build functional model
            visible = Input(shape=(3, in_length))
            hidden1 = LSTM(32, return_sequences=True, name='firstLSTMLayer')(visible)
            hidden2 = LSTM(16, name='secondLSTMLayer', return_sequences=True)(hidden1)
            # left branch decides second agent action
            hiddenLeft = LSTM(10, name='leftBranch')(hidden2)
            agent2 = Dense(5, activation='softmax', name='agent2classifier')(hiddenLeft)
            # right branch decides third agent action
            hiddenRight = LSTM(10, name='rightBranch')(hidden2)
            agent3 = Dense(5, activation='softmax', name='agent3classifier')(hiddenRight)

            model = Model(inputs=visible, outputs=[agent2, agent3])

            model.compile(optimizer='adam',
                          loss={'agent2classifier': 'categorical_crossentropy',
                                'agent3classifier': 'categorical_crossentropy'},
                          metrics={'agent2classifier': ['acc'],
                                   'agent3classifier': ['acc']})

            model.summary()

            U.initialize()

            if fname is not None:
                model.load_weights(fname)

        return model

    elif model_type == "transition":
        visible = Input(shape=(3, in_length))
        hidden1 = LSTM(100, return_sequences=True)(visible)
        hidden2 = LSTM(64, return_sequences=True)(hidden1)
        hiddenObservation = LSTM(64, name='observationBranch')(hidden2)
        observation = Dense(in_length - 3, name='observationScalar')(hiddenObservation)

        # model = Model(inputs=visible,outputs=[agent1,agent2,agent3,observation])
        model = Model(inputs=visible, outputs=observation)
        model.compile(optimizer='adam',
                      loss={'observationScalar': 'mse'},
                      metrics={'observationScalar': ['mae']})

        model.summary()

        U.initialize()

        if fname is not None:
            model.load_weights(fname)

        return model

    elif model_type == "policy":

        with tf.variable_scope(scope):
            # build functional model
            visible = Input(shape=(3, in_length))
            hidden1 = LSTM(32, return_sequences=True, name='firstLSTMLayer')(visible)
            hidden2 = LSTM(16, name='secondLSTMLayer', return_sequences=True)(hidden1)

            hidden_final = LSTM(10, name='leftBranch')(hidden2)
            agent0 = Dense(5, activation='softmax', name='agent0classifier')(hidden_final)

            model = Model(inputs=visible, outputs=agent0)

            model.compile(optimizer='adam',
                          loss={'agent0classifier': 'categorical_crossentropy'},
                          metrics={'agent0classifier': ['acc']})

            model.summary()

            U.initialize()

            if fname is not None:
                model.load_weights(fname)

        return model

    else:
        raise NotImplementedError


def run_agnostic_agent(arglist, threshold=0, attack_rate=1, test=False):
    """
    Black box prediction-based attack
    """

    tf.reset_default_graph()

    scenario = arglist.scenario
    env = make_env(scenario, benchmark=True)

    marl_sess = U.single_threaded_session()

    attacking = True
    timed_attack = False
    attack_threshold = threshold  # Change this to corresponding threshold value

    if scenario == 'simple_spread':
        in_length = 54
        cor_agent_in_length = 18
    else:
        in_length = 28
        cor_agent_in_length = 10

    if scenario == "simple_adversary":
        other_act_model_fname = "Prediction_weights/actionMultiClassNetwork_simpadv"
        transition_model_fname = "Prediction_weights/StateTransitionNetwork_adv"
        corrupted_act_model_fname = "Prediction_weights/adv_agent1_policy_predictor"
    else:
        other_act_model_fname = "Prediction_weights/actionMultiClassNetwork"
        transition_model_fname = "Prediction_weights/StateTransitionNetwork"
        corrupted_act_model_fname = "Prediction_weights/agent0_policy_predictor"

    with marl_sess as sess:

        other_act_model = build_model("Other_prediction", in_length + 1, fname=other_act_model_fname,
                                      model_type="other_policy")
        # U.initialize()
        # act_model.load_weights("Prediction_weights/actionMultiClassNetwork")

        transition_model = build_model("Transition", in_length + 3, fname=transition_model_fname,
                                       model_type="transition")
        # U.initialize()
        # transition_model.load_weights("Prediction_weights/TransitionNetwork")

        cor_act_model = build_model("Agent0_prediction", cor_agent_in_length, fname=corrupted_act_model_fname,
                                    model_type="policy")

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
        if scenario == 'simple_spread':
            U.load_state(arglist.load_dir + "nav_policy", saver=trainer_saver)
        else:
            U.load_state(arglist.load_dir + "adv_policy", saver=trainer_saver)

        env = TfEnv(env)

        episode_rewards = [0.0]
        adv_rewards = [0.0]
        coop_rewards = [0.0]
        iter_rewards = []
        train_step = 0
        episode_step = 0

        obs = env.reset()

        predicted_obs = []

        episode_obs = []
        episode_acts = []
        episode_attacks = 0
        avg_episode_attacks = []

        transition = []

        if scenario == 'simple_spread':
            corrupted_agent = 0
        else:
            corrupted_agent = 1

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

            episode_obs.append(np.concatenate(obs).ravel())

            if episode_step <= 2:
                predicted_obs.append(np.concatenate(obs).ravel())


            probs = [agent.action(o) for (o, agent) in zip(obs, trainers)]

            actions = []

            for j in range(len(trainers)):
                actions.append(np.random.choice(np.arange(len(probs[0])), p=probs[j]))


            if episode_step > 2:

                act_obs = predicted_obs[-1][:cor_agent_in_length]
                predicted_correct_probs = trainers[0].action(act_obs)
                predicted_correct_action = np.random.choice(np.arange(len(predicted_correct_probs)), p=predicted_correct_probs)
                actions[0] = predicted_correct_action


                action_inputs = np.concatenate([episode_acts[-2:], np.expand_dims(actions, axis=0)], axis=0)
                next_state = get_logits(transition_model, episode_obs[-3:], in_length, action_inputs,
                                        model_type="transition")

                predicted_obs.append(np.concatenate(next_state).ravel())




            episode_acts.append(actions)  # for prediction network

            new_obs, rew, done, info_n = env.step(actions)



            if arglist.scenario == 'simple_spread':
                collisions += max([info_n['n'][0][1], info_n['n'][1][1], info_n['n'][2][1]]) - 1
                min_dists += info_n['n'][0][2]
                occ_landmarks += info_n['n'][0][3]

            else:
                adv_dist += info_n['n'][0]
                coop_dist += min(info_n['n'][1][2], info_n['n'][2][2])

            # o1 = np.reshape(np.concatenate((obs, [[actions[0]]])))
            # new_obs = np.reshape(np.asarray(new_obs), [1, 54])

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

            # o = np.asarray(obs)
            # o_next = np.asarray(new_obs)
            # o = np.reshape(o, [1, 54])
            # o_next = np.reshape(o_next, [1, 54])

            # transition.append((o, actions[0], actions[1], actions[2], o_next))

            obs = new_obs

            if done or terminal:
                avg_episode_attacks.append(episode_attacks / 22)
                episode_attacks = 0
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

                episode_obs = []
                episode_acts = []
                obs = env.reset()
                episode_step = 0
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

        # print("AVERAGE REWARD: {}".format(sum(episode_rewards) / len(episode_rewards)))

        # print("ATTACK RATE: {}".format(sum(avg_episode_attacks) / len(avg_episode_attacks)))

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
            np.save('phys_decept_blackbox_prediction_{}_test'.format(int(attack_rate * 100)), transition)
        else:
            # np.save('phys_decept_blackbox_prediction_{}'.format(int(attack_rate * 100)), transition)
            pass

        print("Transition positive rate: {}".format(np.sum(transition[:, 5]) / (22 * arglist.num_episodes)))

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
    run_agnostic_agent(arglist)