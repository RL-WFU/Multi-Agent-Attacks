
from scripts.prediction_attack import run_attack as prediction
from scripts.predicted_agent_0_attacks import attack as policy
import argparse



def parse_maddpg_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=200, help="number of episodes")
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
    parser.add_argument("--save-rate", type=int, default=100,
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
    parser.add_argument("--plots-dir", type=str, default="./data/coop_nav_attack/",
                        help="directory where plot data is saved")

    parser.add_argument("--att-benchmark-dir", type=str, default="Attack/benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--att-plots-dir", type=str, default="Attack/learning_curves/",
                        help="directory where plot data is saved")


    return parser.parse_args()

if __name__ == "__main__":
    """
    Run BlackBox attacks for random, strategically timed, and counterfactual reasoning
    Black box attacks use behavioral cloning for policy prediction
    """

    args = parse_maddpg_args()
    #policy(args, 1, True, 1) #Random attack 100% of timesteps. (Random still chooses worst action according to policy)
    #policy(args, .75, True, .75) #Random attack 75% of timesteps
    #policy(args, .5, True, .5)
    #policy(args, .25, True, .25)

    #policy(args, 0, False, 1) #Strategically timed attack, 100% of timesteps.
    #policy(args, 0.67, False, .75) #ST attack only when c = max(p) - min(p) > 0.67. This equates to around a 75% attack rate
    #policy(args, 0.85, False, .5)
    #policy(args, 0.96, False, .25)

    #prediction(args, 0, 1) #Counterfactual attack, 100% of timesteps
    #prediction(args, 0.75, .75)
    #prediction(args, 2, .5) #CF attack only when sum(kl_div) > 2, which equates to around 50% attack rate
    #prediction(args, 3.25, .25)