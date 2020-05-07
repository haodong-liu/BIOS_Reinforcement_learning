#!/usr/bin/zsh

##本脚本用于实现并行运行top与dstat采集命令，并将输出数据重定向到txt文件中。

##設置時間變量
datetime=`date +%Y%m%d_%H%M%S_%N |cut -b1-20`

mkdir ./$datetime
mkdir ./$datetime/alldata

##创建top、dstat和logiccpu文件夹
mkdir ./$datetime/top
mkdir ./$datetime/dstat
mkdir ./$datetime/logiccpu

##并行运行top、dstat和cat命令
for((i=0;i<3;i++))
do
{ 
	if [ $(($i%3)) -eq 0 ]; then
		##运行top采集指令，利用输出重定向将输出数据导入txt文件。
		top  -b -d 1 -c > ./$datetime/top/1.txt
		echo "" > ./$datetime/top/2.txt
		grep 'top - ' -A 5 ./$datetime/top/1.txt >> ./$datetime/top/2.txt
		sed -i 's/up.*load average: / /g' ./$datetime/top/2.txt
		sed -i "s/min/''/g" ./$datetime/top/2.txt
		sed -i ':a;N;$!ba;s/,/ /g;s/+/ /g;s/\n/ /g;s/--/\n/g'  ./$datetime/top/2.txt
		printf "%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t\n"  Time Average1 Average5 Average15 Tasktotal Taskrunning Tasksleeping CpuUs CpuSy Cpuni CpuId CpuWa Memtotal MemFree MemUsed MemBuff Swaptotal SwapFree SwapUsed SwapCached > ./$datetime/alldata/top.txt

		##截取需要的数据
		awk '{print $3"\t"$4"\t"$5"\t"$6"\t"$8"\t"$10"\t"$12"\t"$19"\t"$21"\t"$23"\t"$25"\t"$27"\t"$37"\t"$39"\t"$41"\t"$43"\t"$47"\t"$49"\t"$51"\t"$53"\t"}' ./$datetime/top/2.txt >> ./$datetime/alldata/top.txt
	elif [ $(($i%3)) -eq 1 ]; then
		##运行dstat采集指令
		dstat -d -r --disk-tps --disk-util -n --socket 1 300 > ./$datetime/dstat/1.txt
		sed -n '3,$p' ./$datetime/dstat/1.txt > ./$datetime/dstat/2.txt
		printf "%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t\n"  DiskRead DiskWrite IORead IOWrite TpsRead TpsWirte DiskUtil NetRead NetWrite Tot Tcp Udp Raw Frg > ./$datetime/alldata/dstat.txt
		awk -F '[|]' '{print $1,$2,$3,$4,$5,$6}' ./$datetime/dstat/2.txt >> ./$datetime/alldata/dstat.txt
		sed -i "2,$ s/k/000/g" ./$datetime/alldata/dstat.txt 
	elif [ $(($i%3)) -eq 2 ]; then	
	    ##运行logiccpu采集指令,查看逻辑cpu个数，针对HPC场景
		printf "%-s\t\n" LogicCpu > ./$datetime/alldata/logiccpu.txt
		cat /proc/cpuinfo| grep "processor"| wc -l > ./$datetime/logiccpu/1.txt
		awk '{print $1}' ./$datetime/logiccpu/1.txt >> ./$datetime/alldata/logiccpu.txt
	fi
}&
done
wait
echo end
