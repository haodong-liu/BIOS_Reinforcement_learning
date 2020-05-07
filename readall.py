# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
@Time    : 2019/9/19
@Author  : Liu Hao Dong
@Function:
@Email   :liuhdies@gmail.com
"""
import os
import re


def get_server_param():
    data = os.popen('top -bi -n 1').read()

    top_data = []

    # 截取需要的字符串
    average_index = data.index("load average:")
    swapavail_index = data.index("avail")
    data = data[average_index:swapavail_index]

    # 提取top命令的数值
    value = re.findall(r"\d+\.?\d*", data)

    # # 提取load_average数据
    # top_data["Average1"] = value[0]
    # top_data["Average5"] = value[1]
    # top_data["Average15"] = value[2]
    #
    # # 提取Task数据
    # top_data["Tasktotal"] = value[3]
    # top_data["Taskrunning"] = value[4]
    # top_data["Tasksleeping"] = value[5]
    #
    # # 提取cpu数据
    # top_data["Cpuus"] = value[8]
    # top_data["Cpusy"] = value[9]
    # top_data["Cpuni"] = value[10]
    # top_data["Cpuid"] = value[11]
    # top_data["Cpuwa"] = value[12]
    #
    # # 提取Mem数据
    # top_data["Memtotal"] = value[16]
    # top_data["Memfree"] = value[17]
    # top_data["Memused"] = value[18]
    # top_data["Membuff/cache"] = value[19]
    # # top_data["Memtotal"] = 1.0
    #
    # # 提取Swap数据
    # top_data["Swaptotal"] = value[20]
    # top_data["Swapfree"] = value[21]
    # top_data["Swapused"] = value[22]
    # top_data["Swapavail"] = value[23]
    # # top_data["Swaptotal"] = 1.0
    top_data.append(float(value[0]))    # 0 Average1
    top_data.append(float(value[1]))    # 1 Average5
    top_data.append(float(value[2]))    # 2 Average15
    top_data.append(float(value[3]))  # 3 Tasktotal
    # top_data.append(0.0)
    try:   # 如果除数是0,做异常处理
        top_data.append(float(value[4])/top_data[3])  # 4 Taskrunning
        top_data.append(float(value[5])/top_data[3])  # 5 Tasksleeping
    except Exception:
        top_data.append(0.0)
        top_data.append(0.0)
    top_data.append(float(value[8]))    # 6  Cpuus
    top_data.append(float(value[9]))    # 7  Cpusy
    top_data.append(float(value[10]))   # 8  Cpuni
    top_data.append(float(value[11]))   # 9  Cpuid
    top_data.append(float(value[12]))   # 10 Cpuwa
    top_data.append(float(value[16]))   # 11 Memtotal
    try:
        top_data.append(float(value[17])/top_data[11])   # 12 Memfree
        top_data.append(float(value[18])/top_data[11])   # 13 Memused
        top_data.append(float(value[19])/top_data[11])   # 14 Membuff/cache
    except Exception:
        top_data.append(0.0)
        top_data.append(0.0)
        top_data.append(0.0)

    top_data.append(float(value[20]))   # 15 Swaptotal
    try:
        top_data.append(float(value[21])/top_data[15])   # 16 Swapfree
        top_data.append(float(value[22])/top_data[15])   # 17 Swapused
        top_data.append(float(value[23])/top_data[11])   # 18 Swapavail
    except Exception:
        top_data.append(0.0)
        top_data.append(0.0)
        top_data.append(0.0)

    top_data[11] = 1.0
    top_data[15] = 1.0

    # print(top_data)
    return top_data
# get_server_param()

