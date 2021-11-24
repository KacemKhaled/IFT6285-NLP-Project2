from os import listdir
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

def dev_data_analysis(folder):
    pass


def make_image_distribution(distribution):
    pass



def train_data_analysis(folder):
    total_nb_train_sentences = 0
    lengthes = []

    files = listdir(folder)

    for fn in tqdm(files):
        with open(folder + fn, 'r', encoding="utf8") as f:
            corpus = f.read().rstrip('\n')
            sentences = corpus.split('\n')
            # print('nb of sentences:', len(sentences))
            total_nb_train_sentences += len(sentences)
            lengthes.extend( [len(sent.split()) for sent in sentences] )

    print(f'total nb of train sentences {total_nb_train_sentences}')
    print('lengthes',lengthes )
    df = pd.DataFrame(lengthes, columns=['length'])
    # df.to_csv('lengths_distribution.csv')

    # todo: make histogramme data from df

    make_image_distribution(df)


train_data_folder = 'train data/training-monolingual.tokenized.shuffled/'
dev_data_folder = 'dev data/'


def main():
    train_data_analysis(train_data_folder)
    #dev_data_analysis(dev_data_folder)


if __name__ == '__main__':
    main()

