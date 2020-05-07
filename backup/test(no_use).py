# # -*- coding: utf-8 -*-
# # !/usr/bin/env python3
# """
# @Time    : 2019/9/18
# @Author  : Liu Hao Dong
# @Function:
# @Email   :liuhdies@gmail.com
# """
# import environment
import numpy as np
# import tensorflow as tf
# print(np.random.randint(0,2,16))
# # print(tf.random_normal([1,19], mean = 100))
# sess = tf.Session()
# aa = environment.envHandler()
# print(np.random.randint(0, 2,  16).astype(float))
# print("------------------")
# bb = tf.random_normal([1,1])
# print("bb:",sess.run(bb))
# # print(int(bb))
# cc = np.array([[2,3,4,5,6]])
# dd = tf.constant([[1.0,1.0,1.0]])
# print("dd", sess.run(dd))
# print(sess.run(tf.concat([bb, dd], axis = 1)))
# # import tensorflow as tf
#
# # # Build a graph.
# # a = tf.constant(5.0)
# # b = tf.constant(6.0)
# # c = a * b
# # # Launch the graph in a session.
# # sess = tf.Session()
# # # Evaluate the tensor `c`.
# # print(sess.run(c))
# #
# # print(tf.concat([aa.BIOS, aa.state], axis = 0))


# a = np.random.randint(0, 16, [1, 16])
# max_index = np.where(a == np.max(a))[1][0]  # 返回需要调整概率最大的action，对应位为1表示需要修改
# print(max_index)
# a = np.zeros(a.shape)
# print(a)
# a[0][max_index] = 1
# print(a)
# print(.1)
# a.astype(float)

a = np.array([[0.0000000e+00,0.0000000e+00,9.0931309e-23,8.7489202e-37,0.0000000e+00,  0.0000000e+00,2.7654975e-28,0.0000000e+00,1.7677274e-37,6.3616409e-34,  0.0000000e+00,2.9854585e-31,0.0000000e+00,1.0000000e+00,0.0000000e+00,  0.0000000e+00]])

print(sum(sum(np.log(a+1))))