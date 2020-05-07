#!/bin/bash

#打开另一个terminal，并运行work.sh脚本，实现了自动配置过程中可能需要中途中断循环

gnome-terminal -x bash -c "bash ./work2.sh;exec bash;"

