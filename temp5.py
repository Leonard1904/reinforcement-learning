# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:40:41 2020

@author: jihoon
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:44:51 2020

@author: jihoon
"""

import numpy as np
import copy

state_num = 2
action_num = 2

# value iteration
bias = np.ones([state_num])*(-10)
policy = np.zeros([state_num], dtype = np.int)
reward = np.array([[-2,-0.5],[-1,-3]])
transp = np.array([[[3/4,1/4],[3/4,1/4]],[[1/4,3/4],[1/4,3/4]]])

state_bar = 0
while True:
    # print(bias)
    old_bias = copy.deepcopy(bias)
    max_bias = -np.inf*np.ones(state_num)
    for state in range(state_num):
        for action in range(action_num):
            max_bias[state] = max(max_bias[state], reward[state,action] + np.sum(transp[action,state]*bias))
    bias = max_bias - max_bias[state_bar]
    gain = max_bias[state_bar]
    bias = bias - bias[state_bar]
    if np.sum((bias-old_bias)**2 > 0.000001) == 0:
        break

for state in range(state_num):
    policy[state] = np.argmax(reward[state] + np.reshape(np.matmul(transp[:,state],np.reshape(bias,[state_num,1])),-1))

print('relative value iteration')
print('gain:',gain,'bias:',bias,', policy:',policy)

# policy iteration
bias = np.ones([state_num])*(-10)
policy = np.array([0,1])
reward = np.array([[-2,-0.5],[-1,-3]])
transp = np.array([[[3/4,1/4],[3/4,1/4]],[[1/4,3/4],[1/4,3/4]]])

state_bar = 0
for _ in range(20):
    transp_pi = np.zeros([state_num,state_num])
    for i in range(state_num):
        for j in range(state_num):
            transp_pi[i,j] = transp[policy[i],i,j]
    reward_pi = np.zeros([state_num])
    for i in range(state_num):
        reward_pi[i] = reward[i,policy[i]]
    while True:
        # print(bias)
        old_bias = copy.deepcopy(bias)
        new_bias = np.zeros(state_num)
        for state in range(state_num):
            new_bias[state] = reward_pi[state] + np.sum(transp_pi[:,state]*bias)
        bias = new_bias - new_bias[state_bar]
        gain = new_bias[state_bar]
        if np.sum((bias-old_bias)**2 > 0.000001) == 0:
            break

    for state in range(state_num):
        policy[state] = np.argmax(reward[state] + np.reshape(np.matmul(transp[:,state],np.reshape(bias,[state_num,1])),-1))

print('policy iteration')
print('gain:',gain,'bias:',bias,', policy:',policy)