import numpy as np
# import pandas as pd
import os
import re

path_train = 'aclImdb_v1/train'
path_test = 'aclImdb_v1/test'

# arrays to append splits

pos_arr, neg_arr = [], []

# read files

for x in os.listdir(path_train):
    if not x == 'unsup':
        if x == 'pos':
            pos_path = os.path.join(path_train, x)
            for pos in os.listdir(pos_path):
                file_pos_path = os.path.join(pos_path, pos)
                with open(file_pos_path, 'r') as p:
                    data = p.readlines()
                    lines = []
                    for line in data:
                        s = line.replace('<br />', '')
                        s = re.sub('[.][.][.]*', '', s)
                        s = re.sub("([.,?!()])", r' \1', s)
                        s = s.strip()
                        lines.append(s)
                        pos_arr.append(lines)
                        lines= []
                        # print(s[:30])
                        # print('\n')

        if x == 'neg':
            neg_path = os.path.join(path_train, x)
            for neg in os.listdir(neg_path):
                file_neg_path = os.path.join(neg_path, neg)
                with open(file_neg_path, 'r') as p:
                    data = p.readlines()
                    for line in data:
                        s = line.replace('<br />', '')
                        s = re.sub('[.][.][.]*', '', s)
                        s = re.sub("([.,?!()])", r' \1', s)
                        # s = s.split()
                        neg_arr.append(s)
                        # print(s[:30])
                        # print('\n')

print(pos_arr[:5])

