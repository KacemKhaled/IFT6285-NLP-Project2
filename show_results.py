import pandas as pd
import argparse
from os import path
import json

## example:
# python show_results.py -n 30 -c -f news trigram

def update_infos():
    if path.exists('results/results.json'):
        with open('results/results.json', 'r') as f:
            scores = json.load(f)
    for model in scores:
        idx = model.find('head-') + len('head-')
        head, backoff = model[idx:idx + 3].split('-')
        head = int(head) if head.isdigit() else 'NaN'
        backoff = int(backoff) if backoff.isdigit() else 0
        scores[model]['head'] = head
        scores[model]['backoff'] = backoff

        if 'loading_time' in scores[model]: scores[model]['loading_time'] = round(scores[model]['loading_time'], 2)
        if 'inference_time' in scores[model]: scores[model]['inference_time'] = round(scores[model]['inference_time'],
                                                                                      2)
    with open('results/results.json', 'w') as f:
        json.dump(scores, f, indent=3)

def get_args():

    parser = argparse.ArgumentParser(description='draft')
    parser.add_argument("-f", '--filter', nargs='+', help="space separated values of budgets", default=[])
    parser.add_argument("-n", '--nb', type=int, help="# files read",default=20)
    parser.add_argument("-c", '--clean', action='store_true', help="remove clomuns",default=False)
    parser.add_argument("-u", '--update', action='store_true', help="update head and backoff columns",default=False)

    return parser.parse_args()

def best_scores():
    args = get_args()
    if args.update:
        update_infos()
    scores = df = pd.read_json('results/results.json')
    df = scores.T.sort_values(by=['bleu_score'],ascending=False)
    #
    df = df.reset_index()
    df = df.rename(columns={"index": "model"})

    for filter in args.filter:
        df = df[df['model'].str.contains(filter)]
    if args.clean:
        df = df.drop(['avg_hamming' , 'avg_norm_hamming' , 'avg_LCSSeq', 'avg_LCSStr'], axis=1)

    print(df.head(args.nb))

best_scores()