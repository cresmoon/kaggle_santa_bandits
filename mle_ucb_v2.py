import numpy as np
import random
from kaggle_environments.envs.mab.mab import Observation, Configuration

EPSILON = 10**(-4)
GRID_SIZE = 10**4
UCB_C = 0.1


class MLEUCBAgentV2:
    def __init__(self, config: Configuration):
        self.agent_name = self.__class__.__name__
        self.num_steps = config.episode_steps
        self.num_arms = config.bandit_count
        self.alpha = config.decay_rate
        
        self.arm_total_n = [0] * self.num_arms
        self.arm_self_n = [0] * self.num_arms
        self.arm_oppo_n = [0] * self.num_arms
        
        self.net_arm_reward = [0] * self.num_arms
        self.total_reward = 0
        
        self.theta = [EPSILON] * self.num_arms
        self.curr_theta = [EPSILON] * self.num_arms
        self.theta_range = np.linspace(EPSILON, 1 - EPSILON, GRID_SIZE - 1)
        self.theta_range_len = len(self.theta_range)
        self.likelihood = [None] * self.num_arms

        self.first_phase_arm = list(range(self.num_arms))
        random.shuffle(self.first_phase_arm)
        self.first_phase_arm_idx = 0
        self.is_first_phase = True

    def add_reward(self, arm, x, n_x):
        if int(x) == 0:
            likelihood_addon = np.log(1 - self.theta_range * self.alpha**n_x)
        else:
            likelihood_addon = np.log(self.theta_range * self.alpha**n_x)
        if self.likelihood[arm] is None:
            self.likelihood[arm] = np.zeros(self.theta_range_len)
        self.likelihood[arm] += likelihood_addon

    def get_theta_mle(self, arm):
        theta_idx = np.argmax(self.likelihood[arm])
        return self.theta_range[theta_idx]

    def __call__(self, obs: Observation):
        net_reward = obs.reward - self.total_reward
        self.total_reward = obs.reward
        t = obs.step + 1
        
        last_arm = None
        for (i, a) in enumerate(obs.last_actions):
            self.arm_total_n[a] += 1
            if i == obs.agent_index:
                self.arm_self_n[a] += 1
                last_arm = a
            else:
                self.arm_oppo_n[a] += 1

        if last_arm is not None:
            last_arm_n = self.arm_total_n[last_arm]
            self.net_arm_reward[last_arm] += net_reward
            self.add_reward(last_arm, net_reward, last_arm_n-1)
            # can optimize a bit here
            self.theta[last_arm] = self.get_theta_mle(last_arm)
            self.curr_theta[last_arm] = self.theta[last_arm] * (self.alpha)**(last_arm_n)

        # naive exploitation
        if (last_arm is not None) and net_reward > 0:
            act = last_arm
            return act

        if self.is_first_phase:
            # still in the first exploration phase
            act = self.first_phase_arm[self.first_phase_arm_idx]
            self.first_phase_arm_idx += 1
            if self.first_phase_arm_idx >= self.num_arms:
                # end of first phase
                self.is_first_phase = False
            return act

        # UCB phase
        c = UCB_C
        max_ucb = 0.0
        max_action = -1
        for (i, theta_i) in enumerate(self.curr_theta):
            delta_i = np.sqrt(c * np.log(t) / self.arm_self_n[i])
            ucb = theta_i + delta_i
            if ucb > max_ucb and last_arm != i:
                max_ucb = ucb
                max_action = i
        if max_action == -1:
            print('Something is seriously wrong, return a random action!')
            max_action = random.randrange(self.num_arms)
        return max_action


cached_agent = None


def agent(raw_observation, raw_configuration):
    global cached_agent
    if cached_agent is None:
        configuration = Configuration(raw_configuration)
        cached_agent = MLEUCBAgentV2(configuration)

    observation = Observation(raw_observation)
    return cached_agent(observation)
