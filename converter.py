"""
author:skanda
"""

import os
import gzip
import csv

home = '/home/skanda/Bouy_Data/cluster_1/2017'
lst = os.listdir(home)
for l in lst:	
	if 'data_meta' in l or '.txt' in l or 'csv_files' in l:
		continue
	file = os.path.join(home,l)
	print file 
	if os.path.exists(file + '.txt'):
		print "yes"
		continue
	with gzip.open(file, 'rb') as f:
		file_content = f.read()
	with open(file + '.txt', "wb") as w:
		w.write(file_content)

print 'DONE TEXT'

os.chdir('../')
file_list = os.listdir("./")
for file in file_list:
	if '.txt' in file:
		csv_file = file.split('.')[0] + '.csv'
		out_file = os.path.join('./csv_files',csv_file)
		if os.path.exists(out_file):
			continue
		with open(file, "rb") as fin:
			lines = fin.readlines()
		out_csv = open(out_file, 'wb')
		output = ""
		for line in lines:	
			line_array = line.split()
			print line_array
			output = (',').join(line_array) + '\n'
			out_csv.write(output)
		out_csv.close()

