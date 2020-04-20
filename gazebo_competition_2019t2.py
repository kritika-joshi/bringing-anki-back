#!/usr/bin/env python
import gym
from gym import wrappers
import gym_gazebo
import time
import numpy
import random
import time

import qlearn
import liveplot

from matplotlib import pyplot as plt

def render():
    render_skip = 0  # Skip first X episodes.
    render_interval = 1000  # Show render Every Y episodes.
    render_episodes = 100  # Show Z episodes every rendering.

    if (x % render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif (((x-render_episodes) % render_interval == 0) and (x != 0) and
          (x > render_skip) and (render_episodes < x)):
        env.render(close=True)

if __name__ == '__main__':

    #need to change this
    env = gym.make('Gazebo_Competition_2019t2-v0')

    outdir = '/tmp/gazebo_gym_experiments'
    env = gym.wrappers.Monitor(env, outdir, force=True)
    #plotter = liveplot.LivePlot(outdir)

    last_time_steps = numpy.ndarray(0)

    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=0.2, gamma=0.8, epsilon=0.9)

    #try:
    qlearn.loadQ("QValues")
    initial_epsilon = qlearn.epsilon
    print("loaded q! its working")
    # except:
    #     initial_epsilon = 0.9
    #     print("did not load q correctly :(")

    epsilon_discount = 0.9986

    start_time = time.time()
    total_episodes = 50
    highest_reward = 0
    max_reward = -3000
    print(" in gazebo comp 2019 t2")
    

    for x in range(total_episodes):
        done = False

        cumulated_reward = 0  # Should going forward give more reward then L/R?

        observation = env.reset()

        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # render() #defined above, not env.render()

        state = ''.join(map(str, observation))

        i = -1
        while True:
            i += 1
            # print(i)

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward
            #print("cumulated reward is")
            #print(cumulated_reward)

            # if highest_reward < cumulated_reward:
            #     highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            qlearn.learn(state, action, reward, nextState)

            env._flush(force=True)
            #print(done)
            # if(i < 10000):
            #     state = nextState
            # else:
            #     last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
            #     break

            # print("main")
            # print(done)
            if not(done):
                state = nextState
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break



        if (cumulated_reward >= highest_reward):
            qlearn.saveQ("QValues")
            highest_reward = cumulated_reward
            print("saved the Q's")  
            #plotter.plot(env)

        print("===== Completed episode {}".format(x))
        print("cumulated reward is:")
        print(cumulated_reward)
        print("highest reward is:")
        print(highest_reward)


        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("Starting EP: " + str(x+1) +
               " - [alpha: " + str(round(qlearn.alpha, 2)) +
               " - gamma: " + str(round(qlearn.gamma, 2)) +
               " - epsilon: " + str(round(qlearn.epsilon, 2)) +
               "] - Reward: " + str(cumulated_reward) +
               "     Time: %d:%02d:%02d" % (h, m, s))

    # Github table content
    print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|" +
           str(qlearn.gamma)+"|"+str(initial_epsilon)+"*" +
           str(epsilon_discount)+"|"+str(highest_reward) + "| PICTURE |")

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".
          format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
