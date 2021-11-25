from os import listdir
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dev_file_analysis(file):
    with open(file, 'r', encoding="utf8") as f:
        corpus = f.read().rstrip('\n')
        sentences = corpus.split('\n')
        print(f'nb of sentences: {len(sentences)} in file {f}')
        lengthes = [len(sent.split()) for sent in sentences]
        df = pd.DataFrame(lengthes, columns=['length'])

    return df


def leng_distribution(df):
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

    lengths_per_cat = [a, b, c, d, e, f, g, h, i, j, k]
    categories = ["<5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45-50", ">=50"]

    return categories, lengths_per_cat


def dev_data_analysis(folder):

    df_news = dev_file_analysis(news_file)
    df_euro = dev_file_analysis(euro_file)
    df_hans = dev_file_analysis(hans_file)

    categories, lengths_per_cat_news = leng_distribution(df_news)
    categories, lengths_per_cat_euro = leng_distribution(df_euro)
    categories, lengths_per_cat_hans = leng_distribution(df_hans)

    make_image_dev_distribution(categories, lengths_per_cat_news, lengths_per_cat_hans, lengths_per_cat_euro)


def make_image_dev_distribution(X, y_news, y_hans, y_euro):
    print('creating image .. ')

    X_axis = np.arange(len(X))

    fig = plt.figure(figsize=(7, 4))
    plt.bar(X_axis - 0.2, y_news, 0.2, label='News')
    plt.bar(X_axis, y_hans, 0.2, label='Hans')
    plt.bar(X_axis + 0.2, y_euro, 0.2, label='Euro')

    plt.xticks(X_axis, X)
    plt.xlabel("Sentences lengths (words)")
    plt.ylabel("Number of sentences")
    #plt.title("Distribution of the dev sentences lengths")
    plt.legend()
    #plt.show()

    fig.savefig('plots/length_distribution_dev.png')
    fig.savefig('plots/length_distribution_dev.eps')
    fig.savefig('plots/length_distribution_dev.pdf')


def make_image_train_distribution(df, name):
    categories, lengths_per_cat = leng_distribution(df)

    print('creating image .. ')
    fig = plt.figure(figsize=(7, 4))
    plt.bar(categories, lengths_per_cat, color='blue', width=0.4)

    plt.xlabel("Sentences lengths (number of words)")
    plt.ylabel("Number of sentences")
    #plt.title("Distribution of the train sentences lengths")
    plt.show()

    fig.savefig('plots/length_distribution_'+name+'.png')
    fig.savefig('plots/length_distribution_'+name+'.eps')
    fig.savefig('plots/length_distribution_'+name+'.pdf')


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
    #
    # df = pd.read_csv('lengths_distribution.csv')
    # df = df.drop(columns=['Unnamed: 0'])
    # make_image_train_distribution(df, 'train')

    dev_data_analysis(dev_data_folder)


if __name__ == '__main__':
    main()

