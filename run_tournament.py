from kaggle_environments import make
from kaggle_environments import evaluate
import ray

ray.init()

NUM_EPISODES = 100


@ray.remote
def run_game(a0, a1, num_episodes):
    env = make('mab', debug=True)
    total_score = [0, 0]
    for _ in range(num_episodes):
        last_state = env.run([a0, a1])[-1]
        reward = [state.reward for state in last_state]
        if reward[0] > reward[1]:
            total_score[0] += 2
        elif reward[0] < reward[1]:
            total_score[1] += 2
        else:
            total_score[0] += 1
            total_score[1] += 1
    return {a0: total_score[0], a1: total_score[1]}


agent_list = [
    'mle_ucb_v3.py',
    'mle_ucb_v2.py',
    'mle_ucb_v1.py',
    'decayed_ucb.py',
    'naive_ucb.py'
]

agent_score = {a: 0 for a in agent_list}
num_tour_agents = len(agent_list)

agent2score = []
for i in range(num_tour_agents):
    for j in range(i+1, num_tour_agents):
        a0, a1 = agent_list[i], agent_list[j]
        print('Evaluate', a0, 'vs.', a1)
        agent2score.append(run_game.remote(a0, a1, NUM_EPISODES))

for a2s in ray.get(agent2score):
    for (a, s) in a2s.items():
        agent_score[a] += s

print(sorted(agent_score.items(), key=lambda x: -x[1]))
