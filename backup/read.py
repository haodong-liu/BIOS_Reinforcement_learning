# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
@Time    : 2019/9/19
@Author  : Liu Hao Dong
@Function:
@Email   :liuhdies@gmail.com
"""
# -*- coding: utf-8 -*-
import os
import re
def get_server_param():
    data = os.popen('top -bi -n 1').read().replace(',','').split()
    top_data = {}
    data_look = os.popen('top -bi -n 1').read()
    average_index = data.index("average:")
    top_data["Average1"] = data[average_index+1]
    top_data["Average5"] = data[average_index+2]
    top_data["Average15"] = data[average_index+3]

    tasktotal_index = data.index("Tasks:")
    top_data["Tasktotal"] = float(data[tasktotal_index+1])
    top_data["Taskrunning"] = float(data[tasktotal_index+3])/top_data['Tasktotal']
    top_data["Tasksleeping"] = float(data[tasktotal_index+5])/top_data['Tasktotal']

    cpu_index= data.index("%Cpu(s):")
    top_data["Cpuus"] = data[cpu_index+1]
    top_data["Cpusy"] = data[cpu_index+3]
    top_data["Cpuni"] = data[cpu_index+5]
    top_data["Cpuid"] = data[cpu_index+7]
    top_data["Cpuwa"] = data[cpu_index+9]

    mem_index = data.index("Mem")

    top_data["Memtotal"] = float(re.findall(r"\d+\.?\d*", data[mem_index+2])[0])
    top_data["Memfree"] = float(data[mem_index+4])/top_data["Memtotal"]
    top_data["Memused"] = float(data[mem_index+6])/top_data["Memtotal"]
    top_data["Membuff/cache"] = float(data[mem_index+8])/top_data["Memtotal"]
    top_data["Memtotal"] = 1.0


    swap_index = data.index("Swap:")
    top_data["Swaptotal"] = float(data[swap_index+1])
    top_data["Swapfree"] = float(data[swap_index+3])/top_data["Swaptotal"]
    top_data["Swapused"] = float(data[swap_index+5])/top_data["Swaptotal"]
    top_data["Swapavail"] = float(data[swap_index+7].split('+')[0])/top_data["Swaptotal"]
    top_data["Swaptotal"] = 1.0
    # print(top_data)
    print(list(top_data.values()))
    return list(top_data.values())


get_server_param()
# print(numpy.array([get_server_param()]).shape)