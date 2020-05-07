#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 本文件作用：读取标记值，修改标记值，修改配置文件

import random
import time 

flag_path = './flag.txt'     # 标志文件路径
config_path = './config.txt'     # bios配置文件路径
label_path = './label.txt'     # label文件路径
time_path = './time.txt'     # time文件路径


# 定义函数，返回整数的某1位的值
def valueAtBit(num, bit):
    return (num >> (bit -1)) & 1;


# 定义函数，返回整数的某2位的值
def valueAt2Bit(num, bit):
    return (num >> (bit -1)) & 3;

# 定义函数，返回整数的某3位的值
def valueAt3Bit(num, bit):
    return (num >> (bit -1)) & 7;
   
#读取标记值
flag_file = open(   flag_path, 'r')
flag_str = flag_file.read()
flag_file.close()
flag_num = int(flag_str)
flag_num = flag_num + 1
flag_file = open(flag_path, 'w')
flag_str = str(flag_num)
flag_file.write(flag_str)
flag_file.close()

#记录时间，供测试使用
time_file = open(time_path, 'w')
time_file.write(time.asctime(time.localtime(time.time())))
time_file.close()

#根据随机数修改biso配置文件
Num = random.randint(0,221883)                  #产生0到221883之间的随机整数
print(Num)
label_file = open(label_path,'w')
label_file.write(str(Num))
label_file.close()

config_file = open(config_path, 'w+')

bit1 = valueAtBit(Num,1)
config_file.write('ProcessorEistEnable:'+str(bit1))        #将产生的随机数的二进制左第一位（最低位）对应于BIOS配置项ProcessorEistEnable
config_file.write('\n')
bit2 = valueAtBit(Num,2)
config_file.write('ProcessorCcxEnable:'+str(bit2))
config_file.write('\n')
bit3 = valueAtBit(Num,3)
config_file.write('NumaEn:'+str(bit3))
config_file.write('\n')
bit4 = valueAtBit(Num,4)
config_file.write('DCUStreamerPrefetcherEnable:'+str(bit4))
config_file.write('\n')
bit5 = valueAtBit(Num,5)
config_file.write('DCUIPPrefetcherEnable:'+str(bit5))
config_file.write('\n')
bit6 = valueAtBit(Num,6)
config_file.write('OSWdtEnable:'+str(bit6))
config_file.write('\n')
bit7 = valueAtBit(Num,7)
config_file.write('VTdSupport:'+str(bit7))
config_file.write('\n')
bit8 = valueAtBit(Num,8)
config_file.write('InterruptRemap:'+str(bit8))
config_file.write('\n')
bit9 = valueAtBit(Num,9)
config_file.write('CoherencySupport:'+str(bit9))
config_file.write('\n')
bit10 = valueAtBit(Num,10)
config_file.write('ATS:'+str(bit10))
config_file.write('\n')
bit11 = valueAtBit(Num,11)
config_file.write('PassThroughDma:'+str(bit11))
config_file.write('\n')
bit12 = valueAtBit(Num,12)
config_file.write('PXE1Setting:'+str(bit12))
config_file.write('\n')

bit16 = valueAtBit(Num,13)
config_file.write('SlotPxeDis:'+str(bit16))
config_file.write('\n')

bit17 = valueAt2Bit(Num,14)
if bit17 == 3:
	bit17 = 2
config_file.write('SataCnfigure:'+str(bit17))
config_file.write('\n')
bit19 = valueAt2Bit(Num,16)
if bit19 == 3:
	bit19 = 2
config_file.write('sSataInterfaceMode:'+str(bit19))
config_file.write('\n')
bit21 = valueAt2Bit(Num,18)
if bit21 == 3:
	bit21 = 2
config_file.write('CustomPowerPolicy:'+str(bit21))
config_file.write('\n')

'''
bit20 = valueAt2Bit(Num,20)
config_file.write('OSWdtAction:'+str(bit20))
config_file.write('\n')

bit22 = valueAt3Bit(Num,22)
if bit22 == 7:
	bit22 = 6
config_file.write('OSWdtTimeout:'+str(bit22))
config_file.write('\n')
'''
config_file.close()

