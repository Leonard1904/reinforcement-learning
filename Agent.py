import threading
import gym
import time
import numpy as np

from Network import Network
from scipy.misc import imresize
from scipy.signal import lfilter


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def size(self):
        return len(self.states)


class Agent(threading.Thread):
    save_lock = threading.Lock()
    global_episode = 0
    global_step = 0
    global_max = -21
    global_moving_average = -21
    global_moving_update = 0

    def __init__(self, name,
                 game, state_size, action_size,
                 global_net, _sess,
                 args,
                 feature_layers=None, critic_layers=None, actor_layers=None
                 ):
        super(Agent, self).__init__()
        self.args = args
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.sess = _sess
        self.global_net = global_net
        self.env = gym.make(game)
        self.local = Network(name, state_size, action_size, global_net.optimizer,
                             feature_layers, critic_layers, actor_layers)
        self.copy_to_local_op = self.local.transfer_weights('global', self.name)
        self.mem = Memory()

    # def _discounted_reward(self, rewards):
    #     return lfilter([1], [1, -self.args.gamma], x=rewards[::-1])[::-1]

    def _discounted_reward(self, rewards):
        return lfilter([1], [1, -self.args.gamma], x=rewards[::-1])

    def act_post_func(self, a):
        return a + 1

    def _preprocess(self, image, height_range=(35, 193), bg=(144, 72, 17)):
        image = image[height_range[0]:height_range[1], ...]
        image = imresize(image, (self.state_size[0], self.state_size[1]), interp="nearest")

        H, W, _ = image.shape

        R = image[..., 0]
        G = image[..., 1]
        B = image[..., 2]

        cond = (R == bg[0]) & (G == bg[1]) & (B == bg[2])

        image = np.zeros((H, W))
        image[~cond] = 1

        image = np.expand_dims(image, axis=2)

        return image

    def play_episode(self):
        env, local, mem, args, global_net, sess = self.env, self.local, self.mem, self.args, self.global_net, self.sess

        s, done, step, counting_step, ep_reward, update_count = env.reset(), False, 0, 0, 0, 0
        s = self._preprocess(s)
        state_diff = s
        mem.clear()
        start = time.time()
        while not done:
            a = local.get_action(state_diff, sess)

            s_, r, done, _ = env.step(self.act_post_func(a))
            s_ = self._preprocess(s_)

            ep_reward, step, counting_step = ep_reward + r, step + 1, counting_step + 1
            mem.store(state_diff, a, r)
            state_diff = s_ - s
            s = s_

            if counting_step >= args.update_freq or r != 0 or done:
                values = np.squeeze(local.get_values(mem.states, sess))
                # if mem.rewards[-1] == 0:
                #     bootstrap_value = values[-1]
                # else:
                #     bootstrap_value = 0
                #
                # discounted_reward = self._discounted_reward(mem.rewards + [bootstrap_value])[:-1]
                # advantages = self._discounted_reward(
                #     np.array(mem.rewards) + self.args.gamma * values[1:] - values[:-1]
                # )

                discounted_reward = self._discounted_reward(mem.rewards)
                discounted_reward -= np.mean(discounted_reward)
                discounted_reward /= np.std(discounted_reward) + 1e-8
                advantages = self._discounted_reward(np.array(mem.rewards) - values)
                advantages -= np.mean(advantages)
                advantages /= np.std(advantages) + 1e-8

                # print(np.array(mem.rewards))
                # print(discounted_reward)
                # print(advantages)
                # input()

                v_l, p_l, e_l, total_loss, grad = sess.run(
                    [local.value_loss, local.policy_loss, local.entropy_loss, local.total_loss, local.apply_grads],
                    feed_dict={
                        local.state: np.array(mem.states),
                        local.selected_action: np.array(mem.actions),
                        local.discounted_reward: discounted_reward,
                        local.advantages: advantages,
                    })

                sess.run(self.copy_to_local_op)

                update_count, counting_step = update_count + 1, 0
                mem.clear()

            if done:
                with Agent.save_lock:
                    Agent.global_moving_average = Agent.global_moving_average * .99 + ep_reward * .01
                    episode_time = update_count / (time.time() - start)
                    Agent.global_moving_update = Agent.global_moving_update * .99 + episode_time * .01 \
                        if Agent.global_moving_update != 0 else episode_time
                    print(
                        # f'{Agent.global_episode}|{Agent.global_step:,}/{int(self.args.max_steps):,}|'
                        f'{Agent.global_episode}|{Agent.global_step:,}|'
                        f' Average: {int(Agent.global_moving_average)}|{(self.args.num_agents * Agent.global_moving_update):.2f} up/sec. '
                        f'{self.name} gets {ep_reward} in {step} steps.'
                    )

                    # self.global_net.summary(sess, Agent.global_moving_average, Agent.global_step)
                    self.global_net.summary(sess, Agent.global_moving_average, Agent.global_episode)
                    Agent.global_episode += 1
                    Agent.global_step += step

                    if Agent.global_max < ep_reward:
                        Agent.global_max = ep_reward

        return ep_reward, step, update_count, time.time() - start

    def run(self):
        self.sess.run(self.copy_to_local_op)
        while True:
            # while Agent.global_step < self.args.max_steps:
            self.play_episode()
        print(Agent.global_max)


class AgentPong(Agent):
    def __init__(self, name,
                 game, state_size, action_size,
                 global_net, _sess,
                 args,
                 feature_layers=None, critic_layers=None, actor_layers=None):
        super().__init__(name,
                         game, state_size, action_size,
                         global_net, _sess,
                         args,
                         feature_layers, critic_layers, actor_layers)

    def act_post_func(self, a):
        return a + 1

    def _discounted_reward(self, rewards):
        return lfilter([1], [1, -self.args.gamma], x=rewards[::-1])[::-1]

    def _preprocess(self, image, height_range=(35, 193), bg=(144, 72, 17)):
        image = image[height_range[0]:height_range[1], ...]
        image = imresize(image, (self.state_size[0], self.state_size[1]), interp="nearest")

        H, W, _ = image.shape

        R = image[..., 0]
        G = image[..., 1]
        B = image[..., 2]

        cond = (R == bg[0]) & (G == bg[1]) & (B == bg[2])

        image = np.zeros((H, W))
        image[~cond] = 1

        image = np.expand_dims(image, axis=2)

        return image
