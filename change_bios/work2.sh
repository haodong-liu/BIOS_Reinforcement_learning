#!/bin/bash

#1、先通过判断在限定时间内是否通过键盘输入指定按键决定是否继续循环
#2、后面使用了expect shell实现自动切换到root用户下，获取权限
#3、进入root之后，运行data_score.sh脚本实现数据采集和Linpack场景运行
#4、保存Linpack结果并修改BIOS，重启
aa='a'
var=$(cat ./flag.txt)
read -p  "请在5s内按下a键和enter鍵离开自启动循環：" -t 5 key_value #等待5秒
if [ $key_value == $aa ] 
then 
    echo "次数为$var,你输入了a"                            
elif [ $var -gt 3000 ]
then
	echo "次数为$var,你没有输入a,次数足够，退出并关机"
	# shutdown
else
    	echo "次数为$var,你没有输入a，继续循环"
    	expect -c
	set timeout 6000
	spawn su - root
	expect \"Password:\"
	send \"123456\r\"
	expect \"*#*\" 	
	send \"source /opt/intel/compilers_and_libraries_2018.3.222/linux/bin/compilervars.sh intel64\r\" 
	expect \"*#*\"                                                  
	send \"cd /home/itlab/hpl-2.2/bin/Linux_Intel64/\r\"
	expect \"*#*\" 
	send \" /home/itlab/hpl-2.2/bin/Linux_Intel64/data_score.sh\r\"
	expect \"*#*\" 
	send \" python /home/itlab/bios/result_s.py\r\"
	expect \"*#*\"
	send \" python /home/itlab/bios/config_c.py\r\"
	expect \"*#*\"
	send \"/home/itlab/uniCfg/uniCfg.uniCfg -wf /home/itlab/bios/config.txt\r \"     
	expect \"*#*\"
	send \"echo hello\r\"
	expect \"*#*\"
	send \"reboot\r\"
	expect \"*#*\"
	interact
    #开机自启动进入root模式——>
    #执行跑分程序
    #执行修改bios程序
    #python /home/liangfeng/Desktop/desk/bios/python_pro/test(no_use).py
    #重启
    #reboot
fi
