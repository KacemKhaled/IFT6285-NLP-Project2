import pandas as pd
import argparse
from os import path
import json

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
    choices=['match', 'bleu_score', 'avg_norm_LCSSeq', 'avg_norm_LCSStr', 'loading_time', 'execution_time'])

    return parser.parse_args()


def update_infos(filename):
    if path.exists(filename):
        with open(filename, 'r') as f:
            scores = json.load(f)
    for model in list(scores.keys()):
        idx = model.find('head-') + len('head-')
        try:
            head, backoff, bf_tail = model[idx:idx + 5].split('-')[:3]
        except:
            head, backoff = model[idx:idx + 5].split('-')[:3]
            bf_tail = ""
        head = int(head) if head.isdigit() else 'NaN'
        backoff = int(backoff) if backoff.isdigit() else 0
        bf_tail = True if bf_tail=='t' else False
        scores[model]['head'] = head
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
    df = df.drop([ 'training_time' , 'model_size' , 'model_score' ,'head'  ,'backoff',  'bf_tail'], axis=1)
# df['qr_time'] *= 1000000 
print(f"Number of filtered results: {len(df)}")
print(f"{'-'*130}\nSorted by :\t{sort}\n{'-'*130}")
print(df.head(args.nb))



