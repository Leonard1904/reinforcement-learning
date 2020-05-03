import numpy as np
import copy

num_state = 2
num_action = 2

# value iteration

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
transition = np.array([p_1, p_2])


def value_iteration(s_bar):
    bias = -100 * np.ones([num_state])
    _bias = np.zeros_like(bias)
    while np.sum((bias - _bias) ** 2 > 0.000001) != 0:
        _bias = copy.deepcopy(bias)
        bias_max = np.ones(num_state) * (-np.inf)
        for s in range(num_state):
            for a in range(num_action):
                bias_max[s] = \
                    max(bias_max[s], r[s, a] +
                        np.sum(transition[a, s] * bias))
        bias = bias_max - bias_max[s_bar]
        gain = bias_max[s_bar]
        bias = bias - bias[s_bar]

    policy = [np.argmax(
        r[s] +
        np.reshape(
            np.matmul(transition[:, s], np.reshape(bias, [num_state, 1])),
            -1
        )
    ) for s in range(num_state)]

    return gain, bias, np.array(policy)


def policy_iteration(s_bar):
    policy = np.array([0, 1])
    bias = -100 * np.ones([num_state])
    _bias = np.zeros_like(bias)
    for _ in range(10):
        transition_pi = np.zeros([num_state, num_state])
        for i in range(num_state):
            for j in range(num_state):
                transition_pi[i, j] = transition[pi[i], i, j]
        reward_pi = np.zeros([num_state])

        for i in range(num_state):
            reward_pi[i] = r[i, pi[i]]

        while np.sum((bias - _bias) ** 2 > 0.000001) != 0:
            _bias = copy.deepcopy(bias)
            new_bias = [reward_pi[s] + np.sum(transition_pi[:, s] * bias)
                        for s in range(num_state)]
            new_bias = np.array(new_bias)
            bias = new_bias - new_bias[s_bar]
            gain = new_bias[s_bar]

        policy = [np.argmax(
            r[s] +
            np.reshape(
                np.matmul(transition[:, s], np.reshape(bias, [num_state, 1])),
                -1
            )
        ) for s in range(num_state)]

    return gain, bias, np.array(policy)


if __name__ == '__main__':
    g, h, pi = value_iteration(0)
    print(g, h, pi)

    # pi iteration
    g, h, pi = policy_iteration(0)
    print(g, h, pi)

# -0.75048828125 [ 0.         -0.33300781] [1 0]
# -0.7500000011634711 [ 0.         -0.33333333] [1 0]
