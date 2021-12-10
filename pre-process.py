import spacy

from time import process_time
import sys
import argparse
import random


# ---------------------------------------------
#        gestion ligne de commande
# ---------------------------------------------

def get_args():

    parser = argparse.ArgumentParser(description='cat <text> | python3 pre-process.py')
    parser.add_argument("-v", '--verbosity', type=int, help="increase output verbosity", default=0) 
    parser.add_argument("-l", '--min', type=int, help="min sentence length", default=5) 
    parser.add_argument("-M", '--max', type=int, help="max sentence length", default=100)
    parser.add_argument("-b", '--nb', type=int, help="# sentences read",default=None)
    parser.add_argument("-m", '--model', type=str, help="the model to use", default='en_core_web_sm')
    parser.add_argument("-n", '--no', type=str, help="blanc-separated list of anti-tokens", default=None)
    parser.add_argument("-o", '--out', type=str, help="output file for reference", default=None)
    parser.add_argument("-w", '--lower', action='store_true', help="lowercase output?",default=False)
    
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

for line in sys.stdin:

    sent = line.strip()
    doc = nlp(sent, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

    nb += 1

    if args.verbosity>0 and ((nb % 1000) == 0):
        tic = process_time()
        print(f'sent: {nb} output: {nbo} time: {tic-start:.2f}', file=sys.stderr)

    l = len(doc)    
    if l >= args.min and l <= args.max:

        toks = [tok.text.lower() if args.lower else tok.text for tok in doc]    
        ok = True
        if args.no:
            ftoks = [t for t in toks if t in anti]
            ok = ok and len(ftoks) == 0

        if ok:

            # output the reference
            if args.out:
                print(" ".join(toks), file=out)

            # output the shuffle
            random.shuffle(toks) 
            print(" ".join(toks))

            nbo += 1

            # too many lines (quit)
            if not args.nb is None and nbo >= args.nb:  
                break

if args.out:
    out.close()

if args.verbosity>0:
    print(f'#total time: {tic-start:.2f} sent: {nb} output: {nbo} ref: {args.out}', file=sys.stderr)

 
