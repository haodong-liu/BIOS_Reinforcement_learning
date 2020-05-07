## 1 项目运行
 `python3 BIOS_Reinforcement_0917.py`

## 2 文件说明
### 2.1 Environment.py
该文件定义服务器环境的变化和reward的计算方式，目前修改BIOS的代码还没有完成。需要结合unicfg完成相关代码

### 2.2 BIOS_Reinforcement_0917.py
该文件编写强化学习网络，并执行训练

### 2.3 readall.py
该文件利用`get_server_param()`函数读取服务器的状态，并做归一化，返回以19个状态组成的列表

 

## 3 model文件夹
保存模型

## 4 backup文件夹

里面是一些代码修改之前的版本，留作备用