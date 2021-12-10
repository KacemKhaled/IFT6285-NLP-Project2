from os import listdir
from tqdm import tqdm
from os import listdir, system, path
import random
import numpy as np

# train_data_folder = 'train_data/training-monolingual.tokenized.shuffled/'
#
# files = listdir(train_data_folder)
#
# for fn in tqdm(files[:1]):
#     with open(train_data_folder + fn, 'r', encoding="utf8") as f:
#         corpus = f.readlines()
#         print(corpus[:3])
#         for i,line in enumerate(corpus):
#             corpus[i] = line.strip()
#         print(corpus[:3])
#

# def pos_match(sent,ref):
#     """
#     Score of matching positions of words between two sentences with different order of words
#     :param sent (str): predicted sentence
#     :param ref (str): true order of sentence
#     :return: score (float)
#     """
#     sent_tokens = sent.split(' ')
#     ref_tokens = ref.split(' ')
#     pos_match = sum([1 for i in range((len(sent_tokens))) if sent_tokens[i] == ref_tokens[i]]) / len(sent_tokens)
#     return pos_match
# sent = "why does ? a big to become such have everything issue"
# ref = "why does everything have to become such a big issue ?"
#
# print(pos_match(sent,ref))

taille = 5
for i in range(taille):
    for j in range(taille ):
        if i+j  == taille - 1 or i  == j:
            print("#", end='')
        else:
            print(" ", end='')
        if j == taille-1 :
            print()

# import kenlm
# model = kenlm.Model('models/trigram_p9.arpa')
# bos = True
# eos = True
# print(model.score('this is a sentence .', bos = bos, eos = eos))
# print(model.score('a sentence is this .', bos = bos, eos = eos))
# print(model.score('. is a this sentence', bos = bos, eos = eos))
# print(model.score('. sentence a is this', bos = bos, eos = eos))
# print(model.score('sentence a is this', bos = bos, eos = eos))
# print()
# print(model.perplexity('this is a sentence .'))
# print(model.perplexity('this is a sentence'))
# print(model.perplexity('a sentence is this .'))
# print(model.perplexity('. is a this sentence'))
# print(model.perplexity('. sentence a is this'))
# print(model.perplexity('sentence a is this'))

# folder_short = 'train_data/preprocessed-short/'
#
# all_files = listdir(folder_short)
# test_files = [file for file in all_files if path.splitext(file)[1]=='.test']
# ref_files = [file for file in all_files if path.splitext(file)[1]=='.ref']
# print(path.splitext(all_files[0]))
# print(len(all_files), all_files[:3])
# print(len(test_files), test_files[:3])
# print(len(ref_files), ref_files[:3])