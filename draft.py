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




def lcstr_words(X, Y):
    """ Dynamic Programming implementation of LCS problem
    Code inspired from https://www.geeksforgeeks.org/python-program-for-longest-common-subsequence/
    """
    X = X.split()
    Y = Y.split()
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]




# This code is contributed by Nikhil Kumar Singh(nickzuck_007)


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