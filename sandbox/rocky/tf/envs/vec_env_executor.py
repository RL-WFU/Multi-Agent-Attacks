

import numpy as np
import pickle as pickle
from sandbox.rocky.tf.misc import tensor_utils
import matplotlib.pyplot as plt


class VecEnvExecutor(object):
    def __init__(self, envs, max_path_length):
        self.envs = envs
        self._action_space = envs[0].action_space
        self._observation_space = envs[0].observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.total_rewards = np.zeros(len(self.envs))
        self.max_path_length = max_path_length
        self.done = False
        self.num_iters = 0
        self.reward_tracker = []
        self.avg_reward_tracker = []

    def step(self, action_n):
        all_results = [env.step(a) for (a, env) in zip(action_n, self.envs)]
        obs, rewards, dones, env_infos = list(map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.total_rewards += rewards
        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done:
                self.done = True
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0

        if self.done:
            self.num_iters += 1
            avg_rewards = sum(self.total_rewards) / len(self.envs)
            self.reward_tracker.append(avg_rewards)
            if self.num_iters % 10 == 0:
                r = sum(self.reward_tracker[-10:]) / 10
                self.log_rewards(r)

            print("AVERAGE REWARD: " + str(avg_rewards))
            self.total_rewards = np.zeros(len(self.envs))
            self.done = False

        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(env_infos)

    def reset(self):
        results = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return results

    def log_rewards(self, rewards):
        self.reward_tracker.append(rewards)
        plt.plot(self.reward_tracker)
        plt.savefig("IRL_reward_plot.png")
        plt.clf()

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def terminate(self):
        pass
