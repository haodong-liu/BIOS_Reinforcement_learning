"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
The cart pole example. Policy is oscillated.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters

# MAX_EPISODE = 3000          # 设置强化学习的最大回合数，此处用不着
DISPLAY_REWARD_THRESHOLD = 10  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 100000   # maximum time step in one episode
RENDER = False  # rendering wastes time  此处用不着
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic 快一些，用于打分，反馈于动作，减少随机的动作
EPOSILON = 0.9  # 非随机选取动作的概率

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = 76  #n_features=19, N个时刻应该赋值 N*19
N_A = 16


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")    #n_features=19,表示19个性能指标
        self.a = tf.placeholder(tf.int32, None, "act")
        self.exp_value = tf.placeholder(tf.float32, None, "td_error")         # expect_value for backpropagation

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(                                             # 神经网络的第一层
                inputs=self.s,
                units=304,                                                     # number of hidden units 76*4=304
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),      # weights
                bias_initializer=tf.constant_initializer(0.1),                # biases
                name='l1'
            )
        # 中间可以再加入隐层 
            self.acts_prob = tf.layers.dense(                                 # 神经网络输出层
                inputs=l1,
                units=n_actions,                                              # output units
                activation=tf.nn.softmax,                                     # get a action
                kernel_initializer=tf.random_normal_initializer(0., .1),      # weights
                bias_initializer=tf.constant_initializer(0.1),                # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):                                      # 定义利用真实奖励作为调整参数的依据
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.exp_value)            # advantage (exp_value) guided loss

        with tf.variable_scope('train'):                                      # 定义目标函数，目标是最大化可以得到的最大奖励
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, exp_value):
        s = s[np.newaxis, :]                                                  # 获取新的状态 
        feed_dict = {self.s: s, self.a: a, self.exp_value: exp_value}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)      # 训练网络
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})                    # get probabilities for all actions
        max_index =  np.where(probs == np.max(probs))[0][0]                   # 返回需要调整概率最大的action，对应位为1表示需要修改
        action = np.zeros(probs.shape)
        action[max_index] = 1
        return action


class Critic(object):
    def __init__(self, sess, n_features, n_actions, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.float32, [1, n_actions], "action")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=np.hstack((self.s, self.a)),                               # 将s和a合并输入,横向方式拼接
                units=140,                                                       # number of hidden units
                activation=tf.nn.relu,                                          # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),        # weights
                bias_initializer=tf.constant_initializer(0.1),                  # biases
                name='l1'
            )

            self.l2 = tf.layers.dense(
                inputs=l1,
                units=16,                                                        # output units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),        # weights
                bias_initializer=tf.constant_initializer(0.1),                  # biases
                name='l2'
            )


            self.v = tf.layers.dense(
                inputs=l2,
                units=1,                                                        # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),        # weights
                bias_initializer=tf.constant_initializer(0.1),                  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)


    def learn(self, s, r, s_,a):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_,self.a:a})
        td_error, _ = self.sess.run([self.td_error, self.train_op],{self.s: s, self.v_: v_, self.r: r})
        return td_error

tf.reset_default_graph()  #新加
sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, n_actions=N_A,lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)


s = env.reset()
t = 0
track_r = []
states = [s]
while t < MAX_EP_STEPS:
    #if RENDER: env.render()
    
    if np.raqom.ranf > EPOSILON:                                          # 如果在概率范围外，随机选择一个动作
        a = np.random.randint(0,16,[1,16])
        max_index =  np.where(probs == np.max(a))[0][0]                   # 返回需要调整概率最大的action，对应位为1表示需要修改
        a = np.zeros(probs.shape)
        a[max_index] = 1
    else:
        a = actor.choose_action(s)                                        # 神经网络选择动作

    s_, r, done, info = env.step(a)                                       # 更新环境， s_为新的环境变量 为1*19的向量，r对应我们的reward，done和 info对服务器能效而言没有用处
                                                                                           #需要  定义一个env对象，实现3个方法，第1是读取环境数据，第2个是计算reward方法，当前性能指标减去
                                                                                            #前一个时刻的增益值 env.reset() ，第3个是env.step(a) 更新动作，设置新的BIOS，等待1s后，状态更新
                                                                                           #   
    states.append(s_)                                                   # 保存当前状态
    if done: r = -20                                                      # 判断游戏是否结束

    track_r.append(r)

    critic.learn(s, r, s_, a)          # gradient = grad[r + gamma * V(s_) - V(s)]    用真实的reward去计算误差，反向传播
    actor.learn(s, a, critic.v)     # true_gradient = grad[logPi(s,a) * td_error] 将critic计算出来的value作为反向传播的依据

    s = states[len(states) - N : ]    # 保存N个时刻的状态，此处N=4
    t += 1


    # 100次循环，计算这100次的平均奖励，然后print出来
    if  t %100 == 0:    
        ep_rs_sum = sum(track_r[t-100],track_r[t])/100
        print("episode:", i_episode, "  reward:", int(running_reward))
