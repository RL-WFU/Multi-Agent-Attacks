import tensorflow as tf

#from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv

from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.imitation_learning import AIRLStateAction
from inverse_rl.models.airl_state import AIRL
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts
from rllab.envs.multiagent.make_adversarial_env import *

import maddpg_implementation.maddpg.common.tf_util as U
from maddpg_implementation.maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg_implementation.experiments.test import get_trainers
import argparse


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import joblib


def parse_maddpg_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
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
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")

    parser.add_argument("--att-benchmark-dir", type=str, default="Attack/benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--att-plots-dir", type=str, default="Attack/learning_curves/",
                        help="directory where plot data is saved")


    return parser.parse_args()

def main():
    arglist = parse_maddpg_args()
    env = make_env('simple_spread')

    experts = load_latest_experts('data/coop_nav', n=100, min_return=None)  # Expert trajectories

    marl_sess = U.single_threaded_session()

    with marl_sess as sess:
        obs_shape_n = [env.all_obs_space[i].shape for i in range(env.n)]
        num_adversaries = 0
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        U.initialize()
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir

        U.load_state(arglist.load_dir + "policy")


        good_agents = trainers[1:]


        env = TfEnv(make_env('simple_spread', policy=good_agents, policy_sess=sess))

        #irl_model = AIRLStateAction(env_spec=env.spec, expert_trajs=experts)
        irl_model = AIRL(env=env, expert_trajs=experts)
        policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))

        algo = IRLTRPO(
            env=env,
            policy=policy,
            irl_model=irl_model,
            n_itr=500,
            batch_size=200,
            max_path_length=25,
            discount=0.99,
            store_paths=True,
            discrim_train_itrs=50,
            irl_model_wt=1.0,
            entropy_weight=0.1, # this should be 1.0 but 0.1 seems to work better
            zero_environment_reward=True,
            baseline=LinearFeatureBaseline(env_spec=env.spec)
        )
        policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy")
        policy_saver = tf.train.Saver(policy_vars)
        with rllab_logdir(algo=algo, dirname='data/coop_nav_AIRL'):
            U.initialize()
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for var in vars:
                print(var)

            algo.train()
            print("Saved vars: {}".format(policy_vars))
            policy_saver.save(marl_sess, "data/coop_nav_AIRL/airl_policy_coopnav")







if __name__ == "__main__":
    main()
