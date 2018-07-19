import gym
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import math
import random
from progressbar import ProgressBar


class rl_agent:
    def __init__(self,env,min_states=2,cluster_buffer=1000,discount_factor=0.1,alpha=0.1,dynamic=False):
        #initialise environment

        self.env = gym.make(env)
        self.alpha = alpha
        self.dynamic = dynamic
        self.discount_factor = discount_factor
        #set the step size parameter
        self.step = 0
        #record the initial action space
        self.num_actions = self.env.action_space.n
        #render the environment to see the initial observation space
        self.num_observations = len(self.env.reset())
        #initialise the q table with only two states
        self.q_table = np.zeros([min_states,self.env.action_space.n])

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

        self.rescale_observations(new=True)
        self.cluster_centers = self.create_clusters()
        self.explorations_since_reclustering = 0

    def get_explore_rate(self,epsilon,t):
        return max(epsilon, min(1, 1.0 - math.log10((t+1)/25)))

    def get_learning_rate(self,t):
        return max(self.alpha, min(0.5, 1.0 - math.log10((t+1)/25)))

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

    def recluster(self,rescale=False,verbose=False):
        self.rescale_observations()

        required_states = self.required_states()
        print(required_states)
        new_classifer = KMeans(n_clusters=required_states)
        new_classifer.fit(self.scaled_observation_data)

        #convert the cluster centers
        new_clusters = new_classifer.predict(self.state_classifier.cluster_centers_)

        #replace state classifier
        self.state_classifier = new_classifer

        #change the q table to match the right clusters
        new_q_table = self.q_table

        for i in range(len(self.state_classifier.cluster_centers_)):
            if verbose:
                print("{} integrated into {}".format(i,new_clusters[i]))
            for action in range(self.num_actions):
                new_q_table[new_clusters[i],action] = self.q_table[i,action]

        #replace the q table
        self.q_table=new_q_table
        self.num_states = required_states


    def create_clusters(self,scaled=False):
        self.state_classifier = KMeans(n_clusters=self.num_states)

        if scaled:
            self.state_classifier.fit(self.scaled_observation_data)
        else:
            self.state_classifier.fit(self.observation_data)

        return self.state_classifier.cluster_centers_

    def rescale_observations(self,new=False):
        #create scaler for data
        if new:
            self.scaler = RobustScaler()
            self.scaler.fit(self.observation_data)

        self.scaled_observation_data = self.scaler.transform(self.observation_data)

    def update_q_table(self,action,reward,t):
        if self.dynamic:
            self.q_table[self.state,action] = self.q_table[self.state,action] + self.alpha*(reward - self.discount_factor*np.max(self.q_table[self.state_prime,:]) - self.q_table[self.state,action])
        else:
            self.q_table[self.state,action] = self.q_table[self.state,action] + self.get_learning_rate(t)*(reward - self.discount_factor*np.max(self.q_table[self.state_prime,:]) - self.q_table[self.state,action])

    def decide_action(self):
        return np.argmax(self.q_table[self.state,])

    def define_state(self,observation,scaled=False):
        if scaled:
            observation = self.scaler.transform(np.asarray([observation]))[0]
        return self.state_classifier.predict([observation])[0]

    def practice(self,max_t=200,visible=False,episodes=1,epsilon=0.05,verbose=False,record=False):
        pbar = ProgressBar()

        previous_episode_performance = None

        if record:
            episode_performance = []

        for episode in pbar(range(episodes)):
            self.step = 0
            episode_rewards = 0
            observation = self.env.reset()


            for t in range(max_t):

                if visible:
                    self.env.render()

                self.state = self.define_state(observation)
                #explore vs exploit
                random_float = random.uniform(0,1)
                if random_float >= epsilon:
                    action = self.decide_action()
                else:
                    if verbose:
                        print("Exploring")
                    action = self.env.action_space.sample()

                observation,reward,done,info = self.env.step(action)
                self.step += 1
                self.state_prime = self.define_state(observation)

                episode_rewards += reward

                if done:
                    self.update_q_table(action=action,reward=0,t=t)
                else:
                    self.update_q_table(action=action,reward=episode_rewards,t=t)




                if done:
                    if verbose:
                        print("Episode {} complete with {} steps".format(episode+1,t+1))

                    if record:
                        episode_performance.append({'episode':episode+1,'steps':t+1})
                    """
                    if previous_episode_performance != None:
                        if t+1 <= 0.1*previous_episode_performance:
                            if verbose:
                                print("reclustering")
                            self.recluster(verbose=verbose)


                    if episode > 0:
                        previous_episode_performance = t+1
                    """


                    break

        self.env.close()

        if record:
            return episode_performance
