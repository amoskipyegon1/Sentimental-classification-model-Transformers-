import pathlib
import re
import tensorflow as tf
import tensorflow_text as text

path = "strings_tokenizer"

tokenizers = tf.saved_model.load(path)

tx = ['It ran at the same time as some other programs about school life ,']

tokens = tokenizers.tokenize(tx)

print(tokens.numpy())
