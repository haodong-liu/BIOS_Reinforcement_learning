#!/bin/bash

#实现数据采集和Linpack场景运行的并行执行

for((i=0;i<2;i++))
do
{
	if [$(($i%2)) -eq 0]
	then

	    ./getdata2.sh                   #运行数据采集脚本
	elif [$((i%2)) -eq 1]
	then
	    mpirun -np 4 ./xhpl                                                                             #运行Linpack
	fi
}&
done
wait
echo end
