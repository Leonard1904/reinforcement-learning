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
    # import gym

    base = Base('base', 10)
    john = John('John', 11)

    base.introduce()
    john.introduce()

    # env = gym.make('Pong-v0')
    # print(env.observation_space.shape)
    # print(env.action_space.n)
