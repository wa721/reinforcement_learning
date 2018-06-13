import gym
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import math
import random


env = gym.make('CartPole-v1')
data = None
#some hyperparameters
number_of_states = 1

explore_rate = 0.001

MIN_LEARNING_RATE = 0.1
discount_factor = 0.99
randomness = 200
cluster_not_made = True

#some performance analysis
episodes = []
scores = []
best_score = 0
time_since_last_shakeup = 0


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25.0)))

def re_configure(clustering_model,scaler,data,q_table,new_states):
    #find the current cluster centers
    cluster_centers = clustering_model.cluster_centers_
    #unscale them
    unscaled_centers = pd.DataFrame(scaler.inverse_transform(cluster_centers))

    #make a new scaler
    new_scaler = RobustScaler()
    #fit the new scaler on the data
    hist_obs = new_scaler.fit_transform(data[:,[0,1,2,3]])

    #make a new clustering model, adding an extra observation
    number_of_new_states = len(cluster_centers)+new_states
    new_clustering_model = KMeans(n_clusters=number_of_new_states)
    new_clustering_model.fit(hist_obs)

    new_q_matrix = pd.DataFrame()
    new_q_matrix['state'] = list(range(number_of_new_states))
    new_q_matrix['0'] = 0.0
    new_q_matrix['1'] = 0.0

    for old_state in range(len(unscaled_centers.index)):
        features = np.asarray(unscaled_centers.loc[old_state,:])
        new_state = new_clustering_model.predict([features])[0]

        new_q_matrix.loc[new_state,["0"]] = q_matrix.loc[old_state,["0"]]

    return new_clustering_model,new_scaler,new_q_matrix

#how to call
#clustering_model,obs_scaler,q_matrix = re_configure(clustering_model,obs_scaler,data,q_matrix,1)

def euclidean_distance(a,b):
    a,b = list(a),list(b)
    if len(a) != len(b):
        raise ValueError("incompatible arrays for euclidean distance")
    distance = 0
    for index in range(len(a)):
        distance += (a[index]-b[index])**2

    return distance**0.5

def augment_reward(observation):
    #relative state depends on the critical observations and their absolute position relative to the permissable extremes
    relative_state = [abs(math.radians(observation[2]))/math.radians(12.0),abs(observation[0])/2.4]
    #closer to perfect, the better the reward
    return 1-euclidean_distance([0,0],relative_state)

def q_learn(q_matrix,s_1,s,t):
    #Q(st,at)
    old_value = q_matrix.loc[(q_matrix['state'] == s),[str(action)]]
    #max(for all a)(Q(st+1,a))
    max_future_reward = max(q_matrix.loc[(q_matrix['state'] == s_1),["0"]].values,q_matrix.loc[(q_matrix['state'] == s_1),["1"]].values)
    #attribute the new Q value
    lr = 0.02#get_learning_rate(t)
    q_matrix.loc[(q_matrix['state'] == s),[str(action)]] = old_value + lr*(reward + discount_factor*max_future_reward - old_value)
    #return the matrix
    return q_matrix

data_defined = False
successes = 0

q_matricies = []

for i_episode in range(40):
    observation = env.reset()
    for t in range(200):
        #env.render()

        if data_defined:
            if cluster_not_made:
                obs_scaler = RobustScaler()
                hist_obs = obs_scaler.fit_transform(data[:,[0,1,2,3]])

                clustering_model = KMeans(n_clusters=number_of_states)
                clustering_model.fit(hist_obs)

                print "Centers"
                print clustering_model.cluster_centers_
                print len(clustering_model.cluster_centers_)

                #initialise the q matrix
                q_matrix = pd.DataFrame()
                q_matrix['state'] = list(range(number_of_states))
                q_matrix['0'] = 0.0
                q_matrix['1'] = 0.0
                print q_matrix

                cluster_not_made = False

            else:
                hist_obs = obs_scaler.transform(data[:,[0,1,2,3]])
                clustered_array = clustering_model.predict(hist_obs)


            #scale the observation for clustering.
            observation = obs_scaler.transform(np.array([list(observation)]))[0]
            #define the current state
            state = clustering_model.predict([observation])[0]

            #choose the best option, or randomly select if there's no clear winner.
            if q_matrix.loc[(q_matrix['state'] == state),["1"]].values > q_matrix.loc[(q_matrix['state'] == state),["0"]].values:
                action = 1

            elif q_matrix.loc[(q_matrix['state'] == state),["1"]].values < q_matrix.loc[(q_matrix['state'] == state),["0"]].values:
                action = 0

            else:
                action = env.action_space.sample()


            #all said, if it's time to explore, it's time to explore
            if random.random() < explore_rate:
                action = env.action_space.sample()


            #make the action
            observation, reward, done, info = env.step(action)

            #adjust the reward to be based on proximity to the best state
            #reward = augment_reward(observation)
            reward = 0.001*t + 1

            #learn
            state_plus_1 = obs_scaler.transform(np.array([list(observation)]))[0]
            state_plus_1 = clustering_model.predict([observation])[0]

            q_matrix = q_learn(q_matrix,s=state,s_1=state_plus_1,t=t)
            #update the observation data. Currently serves no purpose, however may use for ANN at some point
            data = np.concatenate((data,np.array([list(observation)+[reward]])),axis=0)

            #If I've seen something I haven't seen before
            #Change this trigger. We're after something that fires if we're in a world view that is too refined.
            #Perhaps save Q tables and revert back to them if learning stagnates as convergence might not be possible
            if len(q_matrix.loc[q_matrix['1']+q_matrix['0']==0.0,:].index) < 2:
                clustering_model,obs_scaler,q_matrix = re_configure(clustering_model,obs_scaler,data,q_matrix,1)
                print "Reconfigured the q table"


        #if the data array does not exist yet because no data exists yet
        else:
            observation, reward, done, info = env.step(env.action_space.sample())
            print "Random action taken"
            #update the observation data. Currently serves no purpose, however may use for ANN at some point
            data = np.array([list(observation)+[reward]])
            data_defined = True


        if done:
            print("Episode finished after {} timesteps".format(t+1))

            episodes.append(i_episode+1)
            scores.append(t+1)

            break



print q_matrix
plt.figure()
plt.plot(episodes,scores)
plt.show()
