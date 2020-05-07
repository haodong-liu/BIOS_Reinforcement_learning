#!/usr/bin/python
# -*- coding: UTF-8 -*-

#将Linpack运行结果输出文件‘HPL.out'zhong的第46行保存到result2.txt文件中

label_path = '/home/itlab/bios/label.txt'     #label文件路径
Out_path = '/home/itlab/hpl-2.2/bin/Linux_Intel64/HPL.out'           #输出文件路径
result_path = '/home/itlab/bios/result2.txt'     #结果文件路径
line_number = 46  

#读取标记值
label_file = open(label_path, 'r')
label_str = label_file.read()
label_file.close()

Ofile = open(Out_path)
result = Ofile.readlines()[line_number]
Rfile = open(result_path,'a+')
Rfile.write(label_str +'   '+result)
Ofile.close()
Rfile.close()

