class Base:

    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.alias = name + '|'

    def introduce(self):
        print(self.name, self.alias, self.age)


class John(Base):

    def __init__(self, name, age):
        age = age + 1
        super().__init__(name, age)
        print('old', self.alias)
        self.alias = self.name + '*'


if __name__ == '__main__':
    from scipy.signal import lfilter
    import numpy as np
    import gym
    import matplotlib.pyplot as plt
    import cv2
    import time

    # base = Base('base', 10)
    # john = John('John', 11)
    #
    # base.introduce()
    # john.introduce()

    env = gym.make('Pong-v0').unwrapped
    print(env.get_action_meanings())
    #
    # s, done = env.reset(), False
    # count = 0
    # while not done:
    #     # env.render()
    #     s_, r, done, _ = env.step(3)
    #     s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
    #     s = cv2.resize(s, (84, 84), interpolation=cv2.INTER_LINEAR)
    #     print(r)
    #     plt.imshow(s)
    #     plt.title(r)
    #     plt.savefig('./temp/{}'.format(count))
    #     count += 1
    #     s = s_

    # selected_action = [
    #     [0, 1, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ]
    # prob = [
    #     [0.2, 0.4, 0.4],
    #     [0.3, 0.5, 0.2],
    #     [0.333, 0.333, 0.333]
    # ]
    #
    # responsible_outputs = np.sum(np.multiply(selected_action, prob), 1)
    # print(responsible_outputs)
    # policy_loss = -np.log(responsible_outputs)
    # entropy = - (prob * np.log(prob))
    # print(policy_loss)
    # print(entropy)
