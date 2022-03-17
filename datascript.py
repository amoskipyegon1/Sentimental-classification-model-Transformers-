import tensorflow as tf
import os
import re


path_train = 'aclImdb_v1/train'
path_test = 'aclImdb_v1/test'

data_arr, data_labels = [], [], [], []

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
                        data_arr.append(lines)
                        lines= []
                        label = 1
                        data_labels.append(label)

        if x == 'neg':
            neg_path = os.path.join(path_train, x)
            for neg in os.listdir(neg_path):
                file_neg_path = os.path.join(neg_path, neg)
                with open(file_neg_path, 'r') as p:
                    data = p.readlines()
                    lines = []
                    for line in data:
                        s = line.replace('<br />', '')
                        s = re.sub('[.][.][.]*', '', s)
                        s = re.sub("([.,?!()])", r' \1', s)
                        s = s.strip()
                        lines.append(s)
                        data_arr.append(lines)
                        lines = []
                        label = 0
                        data_labels.append(label)



dataset = tf.data.Dataset.from_tensor_slices((data_arr, data_labels))

print('\n')
print(dataset)

