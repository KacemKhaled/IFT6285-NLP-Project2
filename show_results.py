import pandas as pd
import argparse
from os import path,listdir
import json
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import palettes


## example:
# python show_results.py -n 30 -c -f news trigram


def get_args():

    parser = argparse.ArgumentParser(description='draft')
    parser.add_argument("-f", '--filter', nargs='+', help="space separated values of budgets", default=[])
    parser.add_argument("-n", '--nb', type=int, help="# files read",default=20)
    parser.add_argument("-c", '--clean', action='store_true', help="remove clomuns",default=False)
    parser.add_argument("-u", '--update', action='store_true', help="update head and backoff columns",default=False)
    parser.add_argument("-j", '--json_file_results', type=str, help="json file for results", default='results/results_final.json')
    parser.add_argument("-s", '--sort_by', type=str, help="sort by", default='bleu_score',
    choices=['match', 'bleu_score', 'avg_norm_LCSSeq', 'avg_norm_LCSStr', 'loading_time', 'execution_time',
    'head_size'  ,'backoff',  'bf_tail'])

    return parser.parse_args()


def update_infos(filename):
    if path.exists(filename):
        with open(filename, 'r') as f:
            scores = json.load(f)
    for model in list(scores.keys()):
        idx = model.find('head-') + len('head-')
        for dataset in ['news','hans','euro']:
            if dataset in model:
                scores[model]['devset'] = dataset
        try:
            head, backoff, bf_tail = model[idx:idx + 5].split('-')[:3]
        except:
            head, backoff = model[idx:idx + 5].split('-')[:3]
            bf_tail = "0"
        head = int(head) if head.isdigit() else 'NaN'
        backoff = int(backoff) if backoff.isdigit() else 0
        # bf_tail = 't' 
        if bf_tail!='t' and bf_tail!='x':
            bf_tail = 'O'
        scores[model]['head_size'] = head
        scores[model]['backoff'] = backoff
        scores[model]['bf_tail'] = bf_tail

        if 'loading_time' in scores[model]:
            scores[model]['loading_time'] = round(scores[model]['loading_time'], 2)
        if 'inference_time' in scores[model]:
            scores[model]['inference_time'] = round(scores[model]['inference_time'], 2)

        if backoff >= head:
            del scores[model]
    with open(filename, 'w') as f:
        json.dump(scores, f, indent=3)


args = get_args()
filename = args.json_file_results
if args.update:
    update_infos(filename)
scores =  pd.read_json(filename)
sort = args.sort_by
# sort = 'match'
df = scores.T.sort_values(by=[sort],ascending=False)
#
df = df.reset_index()
df = df.rename(columns={"index": "model"})
print(f"Number of total results: {len(df)}")
for filter in args.filter:
    df = df[df['model'].str.contains(filter)]
if args.clean:
    df = df.drop(['avg_hamming' , 'avg_norm_hamming' , 'avg_LCSSeq', 'avg_LCSStr', 'avg_pos_match'], axis=1)
    df = df.drop(['head','training_time' , 'model_size' , 'model_score'], axis=1)
    # df = df.drop([ 'training_time' , 'model_size' , 'model_score' ,'head_size'  ,'backoff',  'bf_tail'], axis=1)
# df['qr_time'] *= 1000000 
print(f"Number of filtered results: {len(df)}")
print(f"{'-'*130}\nSorted by :\t{sort}\n{'-'*130}")
print(df.head(args.nb))


def sns_line_plot(ax,d,x,y,variable,fix1,value1,fix2,value2,xlabel='xlabel',ylabel='ylabel'):
    data = d.groupby(fix1).get_group(value1).groupby(fix2).get_group(value2)
    
    sns.scatterplot(ax=ax,data=data, palette='deep',
    x=x, y=y, hue=variable, style='devset',#sizes=0.5, 
    legend=False) 

    value1 = 0 if value1 == 'O' else value1
    value2 = 0 if value2 == 'O' else value2

    ax.set_title(f"Variable: {variable}\n{fix1}={value1}\n{fix2}={value2}", fontsize=10)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=False)
    ax.legend(#title=variable,
    loc='upper left', framealpha=1,#bbox_to_anchor=(-0.5, 0.05),
           fancybox=False, shadow=False,prop={'size': 6},fontsize=6, title_fontsize=8,scatterpoints=1,
           markerscale=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.grid()
    ax.tick_params(axis='x', which='minor')
    

def plot_1_3(d):
    
    fig, (ax1,ax2,ax3)= plt.subplots(1, 3, figsize=(6,2.5))
    # title = f"Impact of our predefined hyperparameters on the BLEU score and the execution time"
    # fig.suptitle(title)
    y_l = 'Exec. time (s)'
    x_l = 'BLEU Score'
    sns_line_plot(ax=ax1,d=d,x='bleu_score',y='execution_time',variable='head_size',
            fix1='backoff',value1=0,fix2='bf_tail',value2='O',xlabel=x_l+"\n(a)",ylabel=y_l)
    sns_line_plot(ax=ax2,d=d,x='bleu_score',y='execution_time',variable='backoff',
            fix1='head_size',value1=4,fix2='bf_tail',value2='O',xlabel=x_l+"\n(b)",ylabel=y_l)
    sns_line_plot(ax=ax3,d=d,x='bleu_score',y='execution_time',variable='bf_tail',
            fix1='head_size',value1=4,fix2='backoff',value2=3,xlabel=x_l+"\n(c)",ylabel=y_l)
    
    # plt.legend()

    fig.tight_layout() 
    figname= 'times'
    plt.savefig(f"plots/{figname}.svg",format="svg")
    plt.savefig(f"plots/{figname}.png", format="png")
    plt.savefig(f"plots/{figname}.eps", format="eps")
    plt.savefig(f"plots/{figname}.pdf", format="pdf")
    # plt.show()

plot_1_3(df)