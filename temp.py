import gym
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
# import imageio
import numpy as np


def _preprocess(self, image, height_range=(35, 193), size=(42, 42)):
    image = image[height_range[0]:height_range[1], ...]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)

    return image / 255.


if __name__ == '__main__':
    env = gym.make('Pong-v0')
    print(env.get_action_meanings())
    # done = False
    # states = []
    # s = env.reset()
    # states.append(s)
    # while not done:
    #     s, _, done, _ = env.step(1)
    #     states.append(s)
    # print(np.array(states).shape)
    # imageio.mimwrite('./animated_from_images.gif', states)
