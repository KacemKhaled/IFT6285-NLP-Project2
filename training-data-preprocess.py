import spacy

from time import process_time
import sys
import argparse
import random
from os import listdir, path, mkdir
from tqdm import tqdm



# ---------------------------------------------
#        gestion ligne de commande
# ---------------------------------------------

def get_args():

    parser = argparse.ArgumentParser(description='python3 training-pre-process.py folder')
    parser.add_argument("-v", '--verbosity', type=int, help="increase output verbosity", default=0)
    parser.add_argument("-l", '--min', type=int, help="min sentence length", default=5)
    parser.add_argument("-M", '--max', type=int, help="max sentence length", default=25)
    parser.add_argument("-m", '--model', type=str, help="the model to use", default='en_core_web_sm')
    parser.add_argument("-o", '--out', type=str, help="output file for reference", default=None)
    parser.add_argument("-n", '--no', type=str, help="blanc-separated list of anti-tokens", default='" - -- # www http')
    parser.add_argument("-w", '--lower', action='store_true', help="lowercase output?",default=False)
    parser.add_argument("-b", '--nb', type=int, help="# sentences read",default=None)

    return parser.parse_args()


# ---------------------------------------------
#        gestion ligne de commande
# ---------------------------------------------

args = get_args()

if args.no:
    anti = args.no.split()
    if args.verbosity>0:
        print(f'anti-tokens: {len(anti)}: {" ".join(anti)}', file=sys.stderr)

out = None
if args.out:
    out = open(args.out, "wt")


nlp = spacy.load(args.model)
tic = start = process_time() 

nb = nbo = 0

train_data_folder = 'train_data/training-monolingual.tokenized.shuffled/'
train_data_out = 'train_data/out/'
if not path.exists(train_data_out):
    mkdir(train_data_out)

files = listdir(train_data_folder)

for fn in tqdm(files[:2]):
    with open(train_data_folder + fn, 'r', encoding="utf8") as f, \
            open(train_data_out + fn + '.ref', 'w', encoding="utf8") as f_out_ref,\
            open(train_data_out + fn + '.test', 'w', encoding="utf8") as f_out_test:
        corpus = f.read()
        lines = corpus.lower().split('\n') if args.lower else corpus.split('\n')
        for line in lines:
            sent = line.strip()
            toks = sent.split()
            nb += 1
            if args.verbosity>0 and ((nb % 1000) == 0):
                tic = process_time()
                print(f'sent: {nb} output: {nbo} time: {tic-start:.2f}', file=sys.stderr)

            l = len(toks)
            if l >= args.min and l <= args.max:

                if args.no:
                    for t in anti:
                        if t in sent:
                            ok = False
                            break
                    else:
                        ok = True
                    # ftoks = [t for t in anti if t in toks]
                    # ok = ok and len(ftoks) == 0

                if ok:

                    # output the reference
                    if args.out:
                        print(sent+'\n', file=out)
                    else:
                        f_out_ref.write(sent+'\n')

                    # output the shuffle
                    random.shuffle(toks)
                    f_out_test.write(" ".join(toks)+'\n')

                    nbo += 1

                    # too many lines (quit)
                    if not args.nb is None and nbo >= args.nb:
                        break

if args.out:
    out.close()

if args.verbosity>0:
    print(f'#total time: {tic-start:.2f} sent: {nb} output: {nbo} ref: {args.out}', file=sys.stderr)

 
