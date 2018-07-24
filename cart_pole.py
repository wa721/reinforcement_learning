from rl_agent import rl_agent
import matplotlib.pyplot as plt
import pandas as pd

agent = rl_agent(env='CartPole-v1',min_states=500,discount_factor=0.99,alpha=0.1,dynamic=False)
print(agent.q_table)

performance = agent.practice(episodes=5000,visible=False,record=True,epsilon=0.05)

print(agent.q_table)


performance = pd.DataFrame(performance)
plt.plot(performance['episode'],performance['steps'],linewidth=0.1)
plt.xlabel('Episode #')
plt.ylabel('Steps completed')
plt.title('Performance')
plt.show()
