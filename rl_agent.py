import gym
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import math
import random

class rl_agent:
    def __init__(self,env,min_states=2,alpha=0.01,cluster_buffer=10):
        #record the initial action space
        self.num_actions = env.action_space.n
        #render the environment to see the initial observation space
        self.env = env
        self.num_observations = len(self.env.reset())
        #initialise the q table with only two states
        self.q_table = np.array([0.0 for i in range(min_states)]*self.num_actions)
        self.q_table = self.q_table.reshape(min_states,self.num_actions)
        #set the experience to zero
        self.experience = 0
        #provide initial observation for historical data
        observation = self.env.reset()
        self.observation_data = np.array([observation])
        #fill out the historical observation data for the minimum number of state clusters
        self.num_states = min_states
        for t in range(self.num_states*cluster_buffer):
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            self.observation_data = np.append(self.observation_data,[observation],axis=0)
            if done:
                self.env.reset()

        self.rescale_observations()
        self.cluster_centers = self.create_clusters()
        self.explorations_since_reclustering = 0

        #set the alpha value
        self.alpha = alpha

    def required_states(self):
        full_rows = 0
        for row in range(self.q_table.shape[0]):
            if self.q_table[row,0] != 0 and self.q_table[row,0] != 0:
                full_rows += 1
        if full_rows == self.q_table.shape[0]:
            return self.num_states + 1
        elif full_rows <= self.q_table.shape[0] - 1:
            return self.num_states - 1
        else:
            return self.num_states

    def recluster(self):
        self.rescale_observations()

        required_states = self.required_states()

        new_classifer = KMeans(n_clusters=required_states)
        new_classifer.fit(self.scaled_observation_data)

        #convert the cluster centers
        new_clusters = new_classifer.predict(self.state_classifier.cluster_centers_)

        #replace state classifier
        self.state_classifier = new_classifer

        #change the q table to match the right clusters
        new_q_table = self.q_table

        for i in range(len(self.state_classifier.cluster_centers_)):
            print("{} integrated into {}".format(i,new_clusters[i]))
            for action in range(self.num_actions):
                new_q_table[new_clusters[i],action] = self.q_table[i,action]

        #replace the q table
        self.q_table=new_q_table
        self.num_states = required_states


    def create_clusters(self):
        self.state_classifier = KMeans(n_clusters=self.num_states)
        self.state_classifier.fit(self.scaled_observation_data)
        return self.state_classifier.cluster_centers_

    def rescale_observations(self):
        #create scaler for data
        self.scaler = RobustScaler()
        self.scaler.fit(self.observation_data)
        self.scaled_observation_data = self.scaler.transform(self.observation_data)

    def update_q_table(self,action,reward):
        self.q_table[self.state,action] = self.q_table[self.state,action] + self.alpha*(reward - self.q_table[self.state,action])

    def decide_action(self):
        chosen_action = None
        q_a = None
        for action in range(self.num_actions):
            if chosen_action is not None:
                if self.q_table[self.state,action] > q_a:
                    chosen_action = action
                    q_a = self.q_table[self.state,action]
                else:
                    None
            else:
                chosen_action = action
                q_a = self.q_table[self.state,action]

        return chosen_action

    def define_state(self,observation):
        scaled_observation = self.scaler.transform(np.asarray([observation]))[0]
        self.state = self.state_classifier.predict([scaled_observation])[0]

    def practice(self,max_t=200,visible=False,episodes=1,exploration_rate=0.05):
        for episode in range(episodes):
            observation = self.env.reset()

            for t in range(max_t):

                if visible:
                    self.env.render()

                self.define_state(observation)
                #explore vs exploit
                if random.uniform(0,1) >= exploration_rate:
                    action = self.decide_action()
                    observation,reward,done,info = self.env.step(action)
                    self.update_q_table(action=action,reward=reward)
                else:
                    print("Exploring")
                    self.explorations_since_reclustering += 1
                    action = self.env.action_space.sample()
                    observation,reward,done,info = self.env.step(action)



                if self.explorations_since_reclustering > int(0.1*self.observation_data.shape[0]):
                    print(self.q_table)
                    print("Updating clusters")
                    print(self.q_table)
                    self.recluster()
                    self.define_state(observation)
                    self.explorations_since_reclustering = 0

                if done:
                    print("Episode {} complete with {} steps".format(episode+1,t+1))

                    break

        self.env.close()

agent = rl_agent(env=gym.make('CartPole-v1'),min_states=10,alpha=0.99)
print(agent.q_table)
agent.practice(episodes=500)
print(agent.q_table)
