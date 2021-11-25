from os import listdir
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dev_data_analysis(folder):
    pass


def make_image_distribution(df):
    # print(list(df['length']))
    # plt.hist(list(df['length']), bins=11)
    # plt.show()

    print('creating image .. ')
    a = len(df[df['length'] < 5])
    b = len(df[(df['length'] >= 5) & (df['length'] < 10)])
    c = len(df[(df['length'] >= 10) & (df['length'] < 15)])
    d = len(df[(df['length'] >= 15) & (df['length'] < 20)])
    e = len(df[(df['length'] >= 20) & (df['length'] < 25)])
    f = len(df[(df['length'] >= 25) & (df['length'] < 30)])
    g = len(df[(df['length'] >= 30) & (df['length'] < 35)])
    h = len(df[(df['length'] >= 35) & (df['length'] < 40)])
    i = len(df[(df['length'] >= 40) & (df['length'] < 45)])
    j = len(df[(df['length'] >= 45) & (df['length'] < 50)])
    k = len(df[df['length'] >= 50])

    lengthes_per_cat = [a, b, c, d, e, f, g, h, i, j, k]
    categories = ["<5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45-50", ">50"]

    fig = plt.figure(figsize=(7, 4))
    plt.bar(categories, lengthes_per_cat, color='blue', width=1)

    plt.xlabel("Sentences lengths (number of words)")
    plt.ylabel("Number of sentences")
    #plt.title("Distribution of the train sentences lengths")
    plt.show()

    fig.savefig('plots/length_distribution_train.png')
    fig.savefig('plots/length_distribution_train.eps')
    fig.savefig('plots/length_distribution_train.pdf')


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
    #print('lengthes',lengthes )
    df = pd.DataFrame(lengthes, columns=['length'])
    df.to_csv('lengths_distribution.csv')


train_data_folder = 'train data/training-monolingual.tokenized.shuffled/'
dev_data_folder = 'dev data/'
news_file = dev_data_folder + 'news.ref'
euro_file = dev_data_folder + 'euro.ref'
hans_file = dev_data_folder + 'hans.ref'


def main():
    # train_data_analysis(train_data_folder)

    df = pd.read_csv('lengths_distribution.csv')
    df = df.drop(columns=['Unnamed: 0'])
    make_image_distribution(df)


    #dev_data_analysis(dev_data_folder)


if __name__ == '__main__':
    main()

