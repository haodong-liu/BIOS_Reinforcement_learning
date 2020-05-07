# !usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:Liu Hao Dong
@file: .environment.py
@version:
@time: 2019/09/17 
@email:liuhdies@gmail.com
@function： 
"""
#import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
# import BIOS_Reinforcement_1
# import config_c.py
import readall

def read_config(file):
    BIOS = []
    with open(file, encoding = 'utf-8', mode = 'r') as f:
        for line in f.readlines():
            BIOS.append(int(line.split(':')[1]))

    return BIOS


class envHandler(object):
    def __init__(self):
        """
        初始化对象，由于每一次动作是在上一次动作的基础上进行修改，那么对象内要保存当前的BIOS设置

        """
        # self.BIOS = read_config('./config.txt')  # BIOS结果存储在config.txt文件中，初始BIOS选择随机生成的
        self.state = self.read()
        self.BIOS = np.random.randint(0, 2, [1, 16]).astype(float)    # [1 1 1 1 0 0 1 1 1 1 1 0 0 0 1 0]
        self.state_ = None
        print("当前BIOS配置为", self.BIOS, "   当前服务器参数是", self.state)

    def read(self):
        """
        读取当前服务器的状态
        :return: 返回
        """
        # 读取服务器状态，生成一个保存状态的向量
        
        # self.state = pd.read_table('top.txt') #读取存储性能参数数据的文件top.txt
        # state = tf.random_normal([1, 19], mean = 100)
        # state = np.random.uniform(0, 1, [1, 19])
        top_data = readall.get_server_param()
        state = np.array([top_data], float)
        return state

    def change_bios(self, a):  # 定义改变bios的函数
        """
        :param a: ActorNetwork output
        :return:
        """
        # 09-18修改过
        change_index = np.where(a == np.max(a))[1][0]
        print("需要修改的位置是：", change_index)
        self.BIOS.astype(int)
        print("修改之前的BIOS是：", self.BIOS)
        temp = self.BIOS[0][change_index]
        self.BIOS[0][change_index] = int(not temp)

        self.BIOS.astype(float)
        print("修改过的BIOS是：", self.BIOS)

    def step(self, a):
        """
        利用新的BIOS值a，在self.BIOS的基础上对BIOS值进行修改，
        计算reward，并更新self.state
        返回新的状态state_，reward
        消耗时间1s
        :param a: 新动作
        :return: state_（新状态）, r（奖励）
        """
        # TODO: 1. bios配置修改
        # TODO：2. 读取新的BIOS值 self.state_ = self.read()
        # TODO：3. 调用self.reward()
        # TODO：4. self.state = self.state_

        self.change_bios(a)   # 根据输出的action以及当前状态的BIOS，输出下一状态的BIOS
        '''
        用unicfg工具使修改后的BIOS生效，即服务器进入下一个状态
        '''
        # TODO:修改完BIOS之后，需要让BIOS生效，这里没有生效代码，所以环境状态基本没有变化，需要加入unicfg的操作
        self.state_ = self.read()   # 读取下一状态的性能参数

        r = self.reward()
        self.state = self.state_
        
        return self.state, r

    def reward(self):
        """
        self.state_ 和 self.state 对比进行reward计算
        :return: r
        """
        # r = (self.state_['Tasksrunning']*0.1+self.state_['Memused']*0.2+self.state_['Cpuus']*0.7) -(self.state['Tasksrunning']*0.1+self.state['Memused']*0.2+self.state['Cpuus']*0.7)
        # FINISH：reward计算已完成
        reward = ((self.state_[0][4] - self.state[0][4]) * 0.1 +\
            (self.state_[0][13] - self.state[0][13]) * 0.2 +\
            (self.state_[0][6] - self.state[0][6]) * 0.7) * 10
        r = np.array(reward).reshape([1, 1])
        return r
        


