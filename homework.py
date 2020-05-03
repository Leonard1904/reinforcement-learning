import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import linprog

num_state = 2
num_action = 2
p_1 = np.array(
    [
        [3 / 4, 1 / 4],
        [3 / 4, 1 / 4]
    ]
)
p_2 = np.array(
    [
        [1 / 4, 3 / 4],
        [1 / 4, 3 / 4]
    ]
)
r = np.array(
    [
        [-2, -0.5],
        [-1, -3]
    ]
)
gamma = 0.9


def operator(reward, transition, value):
    v_ = reward + gamma * np.squeeze(np.matmul(transition, value))
    err = mean_squared_error(v_, value)
    return np.expand_dims(v_, 1), err


def bellman_operator(value):
    temp = [
        r[i] + gamma * np.dot(np.array([p_1[i], p_2[i]]), value)
        for i in range(num_state)
    ]
    return np.max(temp, axis=1), np.argmax(temp, axis=1)


def linear():
    transition = np.array([p_1, p_2])
    c = np.ones(num_state)
    A_ub = np.zeros([num_state * num_action, num_state])
    for state in range(num_state):
        for action in range(num_action):
            for next_state in range(num_state):
                if state == next_state:
                    A_ub[state * num_action + action, next_state] = \
                        gamma * transition[action, state, next_state] - 1
                else:
                    A_ub[state * num_action + action, next_state] = \
                        gamma * transition[action, state, next_state]
    b_ub = np.zeros(num_state * num_action)
    for state in range(num_state):
        for action in range(num_action):
            b_ub[state * num_action + action] = -r[state, action]

    result = linprog(c, A_ub, b_ub, bounds=(-100, 100))
    value = result.x
    return value


if __name__ == '__main__':
    # value iteration
    v = np.array(
        [0.7, 0.3]
    )
    for j in range(100):
        v, pi = bellman_operator(v)
    print(v, pi)

    # policy iteration
    v = np.array(
        [0.7, 0.3]
    )
    for j in range(100):
        v, pi = bellman_operator(v)
        print(v, pi)

    # linear programming
    print(linear())
