from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
import torch
from torch import nn
from transformers import pipeline


#
# tokenizer = AutoTokenizer.from_pretrained("albert-v2-base")
# model = AutoModelForCausalLM.from_pretrained("albert-v2-base")


ref = "why does everything have to become such a big issue ?"
sent_1 = '? everything big why to become does have such issue a'
sent_2 = "a big issue to have become such ? why does everything"
sent_3 = "why does everything have to become such a big ? issue"
sent_4 = "? why does everything have to become such a big issue"

sequence = f"why does"
word = "everything"
from random import randint
from numpy import array
from numpy import argmax
from keras.utils import to_categorical

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]

# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
	X1, X2, y = list(), list(), list()
	for _ in range(n_samples):
		# generate source sequence
		source = generate_sequence(n_in, cardinality)
		# define target sequence
		target = source[:n_out]
		target.reverse()
		# create padded input target sequence
		target_in = [0] + target[:-1]
		# encode
		src_encoded = to_categorical([source], num_classes=cardinality)
		tar_encoded = to_categorical([target], num_classes=cardinality)
		tar2_encoded = to_categorical([target_in], num_classes=cardinality)
		# store
		X1.append(src_encoded)
		X2.append(tar2_encoded)
		y.append(tar_encoded)
	return array(X1), array(X2), array(y)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# configure problem
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
# generate a single source and target sequence
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
print(X1.shape, X2.shape, y.shape)
print('X1=%s, X2=%s, y=%s' % (one_hot_decode(X1[0]), one_hot_decode(X2[0]), one_hot_decode(y[0])))