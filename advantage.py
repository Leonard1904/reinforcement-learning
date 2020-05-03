import numpy as np
from scipy.signal import lfilter


def _discounted_reward(r, g):
    return lfilter([1], [1, -g], x=r[::-1])[::-1]


def gae(r, lamda):
    return _discounted_reward(r, lamda * 0.99)


# def get_advantages(values, masks, rewards, lmbda):
#     returns = []
#     gae = 0
#     for i in reversed(range(len(rewards))):
#         delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
#         gae = delta + gamma * lmbda * masks[i] * gae
#         returns.insert(0, gae + values[i])
#
#     adv = np.array(returns) - values[:-1]
#     return returns, adv


if __name__ == '__main__':
    # gamma = .99
    # rewards = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    # values = np.array([.5, .2, -.2, .3, .4, .4, .6, -.4, .5])
    # delta = rewards + gamma * values[1:] - values[:-1]

    # print(_discounted_reward(rewards, 0.99))
    # print(delta)
    # ret, gae_ = get_advantages(values, [1, 1, 1, 1, 1, 1, 1, 1], rewards, 0.95)
    # print(ret)
    # print(gae_)
    # print(gae(delta, 0.95))

    # a = b = 1
    # b += 1
    # print(a, b)
    # from queue import Queue
    #
    # # Initializing a queue
    # q = Queue(maxsize=3)
    # q.put(1)
    # q.put(2)
    # q.put(3)
    # x = np.array(q.queue)
    # print(x)
    # print(np.array(q.queue))

    import gym

    env = gym.make('PongDeterministic-v0')
    print(env.get_action_meanings())
