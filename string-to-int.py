# import tensorflow_text as text
import tensorflow as tf
# import numpy as np
import time
import os
import re
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

path_train = 'aclImdb_v1/train'

unsup = []

for x in os.listdir(path_train):
    if x == 'unsup':
        unsup_path = os.path.join(path_train, x)
        for pos in os.listdir(unsup_path):
            file_unsup_path = os.path.join(unsup_path, pos)
            with open(file_unsup_path, 'r') as p:
                data = p.readlines()
                lines = []
                for line in data:
                    s = line.replace('<br />', '')
                    s = re.sub('[.][.][.]*', '', s)
                    s = re.sub("([.,?!()])", r' \1', s)
                    lines.append(s.strip().lower())
                    unsup.append(lines)
                    lines = []

# unsup = np.array(unsup)
# print(unsup[:10])

unsup_tensor = tf.data.Dataset.from_tensor_slices(unsup)

bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ['[PAD]', '[UNK]', '[START]', '[END]']

bert_vocab_args = dict(
    vocab_size = 8000,
    reserved_tokens=reserved_tokens,
    bert_tokenizer_params=bert_tokenizer_params,
    learn_params={},
)


text_vocab = bert_vocab.bert_vocab_from_dataset(
    unsup_tensor.batch(64).prefetch(2),
    **bert_vocab_args
)

def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)


write_vocab_file('vocabulary.txt', text_vocab)