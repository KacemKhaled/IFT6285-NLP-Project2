from os import listdir, system, path, mkdir
import time
import kenlm
import argparse
from statistics import mean
from sacrebleu.metrics import BLEU
import textdistance
import numpy as np
import random
from tqdm import tqdm
import json
import itertools
from transformers_utils import GPT2

folder_original = 'train_data/training-monolingual.tokenized.shuffled/'
folder_short = 'train_data/preprocessed-short/'
folder_full = 'train_data/preprocessed-full/'

folder_test = 'train_data/heldout/'
kenlm_dir = '/home/kacem/kenlm/'
# test_file = folder_test+'news.en-00000-of-00100'
model_file = 'models/bigram'



def get_args():

    parser = argparse.ArgumentParser(description='KenLM')
    parser.add_argument("-f", '--folder', type=str, help="the folder to use",choices=['short','full', 'original'], default='full')
    parser.add_argument("-s", '--scoring', type=str, help="the score to use",choices=['perplexity','score'], default='perplexity')
    parser.add_argument("-m", '--model_file', type=str, help="the model to use", default='models/bigram')
    parser.add_argument("-v", '--verbosity', type=int, help="increase output verbosity", default=0)
    parser.add_argument("-n", '--nb', type=int, help="# files read",default=2)
    parser.add_argument("-d", '--head', nargs='+',type=int, help="head size, default : nothing",default=[0])
    parser.add_argument("-b", '--backoff', nargs='+',type=int, help="backoff, default: 1",default=[0])
    parser.add_argument("-o", '--order', type=int, help="2: bigram, 3: trigram",default=2)
    parser.add_argument("-t", '--retrain', action='store_true', help="force retraining",default=False)
    parser.add_argument("-p", '--reprocess', action='store_true', help="force reprocessing",default=False)

    return parser.parse_args()

def find_permutations(seq,r):
    """
    Find the possible r permutations from a sequence
    :param seq (List): sequence of str
    :param r (int): length of sub-sequence: (1,2,3)
    :return permutations (List):
    """
    permutations = []
    n = range(len(seq))
    if r==3:
        permutations = [[seq[i],seq[j],seq[k]] for i in n for j in n for k in n if i != j and i != k and j != k]
    elif r==2:
        permutations = [[seq[i], seq[j]] for i in n for j in n if i != j]
    elif r==1:
        permutations = [[t] for t in seq]
    else:
        permutations = [list(t) for t in itertools.permutations(seq,r)]

    return permutations


def solver_full(model, sentence,head_size = 1,scoring='perplexity',verbose=0,backoff=0):
    """
    Find the most probable ordering of a sequence e of tokens using a greedy search with kenlm
    trigram/bigram for scoring we assume that sequences are longer than 3 words

    :param model (kenlm model):
    :param sentence str:
    :param head_size int: number of words in the starting head of a sentence
    :return: str
    """
    tokens = sentence.split(' ')
    permutations = itertools.permutations(tokens,head_size)
    if verbose:
        print(f"Found {len(permutations)} permutations with subsequence length {head_size} from sequence length: {len(tokens)}")
    best_score = 10000000 #TODO -inf if score else +inf if perplex
    best_sequence = []
    t = time.time()
    for p in permutations:
        p = list(p)
        toks = tokens[:]
        for i in range(head_size):
            toks.remove(p[i])
        new_sequence = p+toks
        if scoring == 'perplexity':
            score = model.perplexity(" ".join(new_sequence)) # TODO
        elif scoring == 'score':
            score = model.score(" ".join(new_sequence), bos=False, eos=False)
        # print(seq,score)
        if score < best_score: #TODO > score or < perplexity
            best_score = score
            best_sequence = new_sequence[:]
    tokens = best_sequence[:]
    i=head_size
    while i < len(tokens):
        best_score = 10000000 ##TODO -inf if score else +inf if perplex
        best_candidate = "UNK"
        the_rest = tokens[i:]
        for t,token in enumerate(the_rest):
            new_sequence = tokens[:i] + [token]+the_rest[:t]+the_rest[t+1:]
            if scoring == 'perplexity':
                score = model.perplexity(" ".join(new_sequence)) #TODO
            elif scoring == 'score':
                score = model.score(" ".join(new_sequence), bos=False, eos=False)
            if score < best_score: ##TODO > score or < perplexity
                best_score = score
                best_sequence = new_sequence[:]
        i+=1
    if verbose: print(f"Prediction time: {time.time()-t:e}s")

    result = " ".join(best_sequence)
    if verbose: print(f"{len(sentence.split())} words\t{len(sentence)} caracters:\t{sentence}\n"
          f"{len(result.split())} words\t{len(result)} caracters:\t{result} ")
    return result


def solver_acc(model, sentence, head_size = 1,scoring='perplexity',verbose=0,backoff=0):
    """
    Find the most probable ordering of a sequence e of tokens using a greedy search with kenlm
    trigram/bigram for scoring
    we assume that sequences are longer than 3 words

    :param model (kenlm model):
    :param sentence str:
    :return: str
    """
    tokens = sentence.split()
    permutations = find_permutations(tokens,head_size)
    if verbose:
        print(f"Found {len(permutations)} permutations with subsequence length {head_size} from sequence length: {len(tokens)}")
    best_score = 10000000
    best_sequence = []
    t = time.time()
    for seq in permutations:
        if scoring == 'perplexity':
            score = model.perplexity(" ".join(seq))
        elif scoring == 'score':
            score = - model.score(" ".join(seq), bos=False, eos=False) # not the enf of sentence
        # print(seq,score)
        if score < best_score:
            best_score = score
            best_sequence = seq[:]
    try:
        assert len(best_sequence) == head_size
    except:
        exit(f"Length mismatch {len(best_sequence) , head_size} \n-tokens: {tokens}\n-best_sequence: {best_sequence}")
    for i in range(head_size):
        tokens.remove(best_sequence[i])
    while tokens:
        best_score = 10000000 #TODO -inf if score else +inf if perplex
        best_candidate = "UNK"
        for i,token in enumerate(tokens):
            if scoring == 'perplexity':
                score = model.perplexity(" ".join(best_sequence+[token])) # TODO
            elif scoring == 'score':
                score = model.score(" ".join(best_sequence+[token]),
                                    bos=False,
                                    eos=False) # eos == False at the end of the sentence
            if score < best_score: ##TODO > score or < perplexity
                best_score = score
                best_candidate = token
        best_sequence.append(best_candidate)
        tokens.remove(best_candidate)
    # print(f"Loading time: {loading_time:.2f}s\tPrediction time: {time.time()-t:e}s")
    if verbose: print(f"Prediction time: {time.time() - t:e}s")
    result = " ".join(best_sequence)
    if verbose: print(f"{len(sentence.split())} words\t{len(sentence)} caracters:\t{sentence}\n"
          f"{len(result.split())} words\t{len(result)} caracters:\t{result} ")
    return result



def solver_acc_2(model, sentence, head_size = 1, scoring='perplexity',verbose=0,backoff=0):
    """
    Find the most probable ordering of a sequence e of tokens using a greedy search with kenlm
    trigram/bigram for scoring
    we assume that sequences are longer than 3 words

    :param model (kenlm model):
    :param sentence str:
    :return: str
    """
    tokens = sentence.split()
    final_sequence = []
    while tokens:
        temp = final_sequence[:]
        if len(tokens) < 8:
            head_size = 7
        permutations = find_permutations(tokens,min(head_size,len(tokens)))
        # permutations = itertools.permutations(tokens,min(head_size,len(tokens)))
        if verbose:
            print(f"Found {len(permutations)} permutations with subsequence length {head_size} from sequence length: {len(tokens)}")
        best_score = 10000000
        best_sequence = []
        t = time.time()
        for seq in permutations:
            if scoring == 'perplexity':
                score = model.perplexity(" ".join(temp+seq))
            elif scoring == 'score':
                score = - model.score(" ".join(temp+seq), bos=False, eos=False) # not the enf of sentence
            # print(seq,score)
            if score < best_score:
                best_score = score
                best_sequence = seq[:]
        for i in range(len(best_sequence)):
            tokens.remove(best_sequence[i])
        final_sequence.extend(best_sequence)

    # print(f"Loading time: {loading_time:.2f}s\tPrediction time: {time.time()-t:e}s")
    if verbose: print(f"Prediction time: {time.time() - t:e}s")
    result = " ".join(final_sequence)
    if verbose: print(f"{len(sentence.split())} words\t{len(sentence)} caracters:\t{sentence}\n"
          f"{len(result.split())} words\t{len(result)} caracters:\t{result} ")
    return result


def solver_acc_3(model, sentence, head_size = 1, scoring='perplexity',verbose=0,backoff=1):
    """
    Find the most probable ordering of a sequence e of tokens using a greedy search with kenlm
    trigram/bigram for scoring
    we assume that sequences are longer than 3 words

    :param model (kenlm model):
    :param sentence str:
    :return: str
    """
    tokens = sentence.split()
    final_sequence = []
    while tokens:
        temp = final_sequence[:]
        if len(tokens) < 8:
            head_size = 7
        # elif len(tokens) < 10:
        #     head_size = 5
        permutations = find_permutations(tokens,min(head_size,len(tokens)))
        if verbose:
            print(f"Found {len(permutations)} permutations with subsequence length {head_size} from sequence length: {len(tokens)}")
        best_score = 10000000
        best_sequence = []
        t = time.time()
        for seq in permutations:
            if scoring == 'perplexity':
                score = model.perplexity(" ".join(temp+seq))
            elif scoring == 'score':
                score = - model.score(" ".join(temp+seq), bos=False, eos=False) # not the enf of sentence
            # print(seq,score)
            if score < best_score:
                best_score = score
                best_sequence = seq[:]
        new_range = len(best_sequence)-backoff if len(best_sequence)>backoff else len(best_sequence)
        for i in range(new_range):
            tokens.remove(best_sequence[i])
        final_sequence.extend(best_sequence[:new_range])

    # print(f"Loading time: {loading_time:.2f}s\tPrediction time: {time.time()-t:e}s")
    if verbose: print(f"Prediction time: {time.time() - t:e}s")
    result = " ".join(final_sequence)
    if verbose: print(f"{len(sentence.split())} words\t{len(sentence)} caracters:\t{sentence}\n"
          f"{len(result.split())} words\t{len(result)} caracters:\t{result} ")
    return result



def dummy_solve(sentence):
    def shuffle_forward(l):
        order = list(range(len(l)))
        random.shuffle(order)
        return list(np.array(l)[order]), order

    def shuffle_backward(l, order):
        l_out = [0] * len(l)
        for i, j in enumerate(order):
            l_out[j] = l[i]
        return l_out
    tokens = sentence.split()
    random.shuffle(tokens)
    l_shuf, order = shuffle_forward(tokens)
    print(" ".join(l_shuf))
    l_unshuffled = shuffle_backward(l_shuf, order)
    result = " ".join(l_unshuffled)
    print(f"{len(sentence.split())} words\t{len(sentence)} caracters:\t{sentence}\n"
          f"{len(result.split())} words\t{len(result)} caracters:\t{result} ")
    return result

def compute_pos_match(sent,ref):
    """
    Score of matching positions of words between two sentences with different order of words
    :param sent (str): predicted sentence
    :param ref (str): true order of sentence
    :return: score (float)
    """
    sent_tokens = sent.strip().split()
    ref_tokens = ref.strip().split()
    # assert len(sent_tokens) == len(ref_tokens)
    try:
        pos_match = sum([1 for i in range((len(sent_tokens))) if sent_tokens[i] == ref_tokens[i]]) / len(sent_tokens)
    except:
        print(f"Length mismtatch {len(sent_tokens), len(ref_tokens)} between: {sent, ref}" )
        pos_match = 0.0
    return pos_match

def lcsseq_words(X, Y):
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

def evaluate(sentences, references):
    """Evalute two lists of strings of same lengths

    :param sentences (List[str]): predicted sentences
    :param references (List[str]): true labels
    :return: metrics (dict)
    """
    print("\nEvaluation..")
    assert len(sentences) == len(references)
    Nb_sent = len(sentences)
    print(f"Nb of sentences : {len(sentences)}")
    # Binary evaluation
    print(sentences[0], references[0],sep='\n')
    b = 0
    for ind in range(len(sentences)):
        if sentences[ind] == references[ind]:
            b = b + 1

    #blue score
    bleu = BLEU()
    bleu_score = bleu.corpus_score(sentences, [references]) # reference should be a list of a list

    # text distance
    d_hamming = [ textdistance.hamming.similarity(sentences[inx], references[inx]) for inx in range(Nb_sent)]
    avg_hamming = mean(d_hamming)

    d_LCSSeq = [lcsseq_words(sentences[inx], references[inx]) for inx in range(Nb_sent) ]
    avg_LCSSeq = mean(d_LCSSeq)

    d_LCSStr = [textdistance.lcsstr.similarity(sentences[inx].split(), references[inx].split())  for inx in range(Nb_sent) ]
    avg_LCSStr = mean(d_LCSStr)

    pos_match = [compute_pos_match(sentences[inx], references[inx]) for inx in range(Nb_sent)]
    avg_pos_match = mean(pos_match)

    ### NOMRALIZATION OF METRICS
    # normalized text distance-> 1.0 exact match, because sequences may have different lengths, so averaging all distances is not fair
    norm_d_hamming = [textdistance.hamming.similarity(sentences[inx], references[inx]) / len(sentences[inx]) for inx in
                      range(Nb_sent)]
    avg_norm_hamming = mean(norm_d_hamming)

    norm_d_LCSSeq = [lcsseq_words(sentences[inx], references[inx]) / len(sentences[inx].split()) for inx in range(Nb_sent)]
    avg_norm_LCSSeq = mean(norm_d_LCSSeq)

    norm_d_LCSStr = [textdistance.lcsstr.similarity(sentences[inx].split(), references[inx].split()) / len(sentences[inx].split()) for inx in
                range(Nb_sent)]
    avg_norm_LCSStr = mean(norm_d_LCSStr)


    metrics = {
        "match" : b / len(sentences),
        "bleu_score" : round(bleu_score.score,4),
        "avg_hamming" : round(avg_hamming,4),
        "avg_norm_hamming" : round(avg_norm_hamming,4),
        "avg_LCSSeq" : round(avg_LCSSeq,4),
        "avg_norm_LCSSeq" : round(avg_norm_LCSSeq,4),
        "avg_LCSStr" : round(avg_LCSStr,4),
        "avg_norm_LCSStr" : round(avg_norm_LCSStr,4),
        "avg_pos_match": round(avg_pos_match,4),
               }
    return metrics


def evaluate_files(dev_file,ref_file):
    with open(ref_file,encoding='utf-8') as ref, \
        open(dev_file,encoding='utf-8') as dev:
        sent_ref = ref.read().split('\n')[:-1]
        sent_dev = dev.read().split('\n')[:-1]
        # print(sent_ref[:2])
        # print(sent_ref[-1])
        # print(sent_dev[:2])
        # print(sent_dev[-1])

        len_sentence = len(sent_ref)
        assert len(sent_ref) == len(sent_dev)
        metrics  = evaluate(sent_dev, sent_ref)


    return metrics



def predict(model,test_file,solve_func,head_size,version="",
            verbose=0,evaluation_only=False,scoring='perplexity',backoff=0):
    dev_file = path.splitext(test_file)[0]+str(version)+'.dev'
    ref_file = path.splitext(test_file)[0]+'.ref'
    print(test_file,dev_file,ref_file)

    if not evaluation_only:
        with open(test_file, encoding='utf-8') as test,\
                open(dev_file,'w',encoding='utf-8') as dev:
            sentences = test.read().split('\n')
            for sent in tqdm(sentences):
                if sent != "":
                    prediction = solve_func(model, sent, head_size,scoring,verbose=verbose,backoff=backoff)
                    dev.write(prediction+'\n')
                    dev.flush()
    score = evaluate_files(ref_file, dev_file)
    print(version,score)
    return score


def preprocess_folder(folder,nb,force=False):
    # creates a folder called 'pre-processed' inside the folder given as input
    # Pre-preocesses all the files inside the input folder
    # Saves the pre-processed files iside the new created folder

    print("Pre-processing files in :", folder)
    out_path = folder[:-1] + '-pre-processed/'

    if not path.exists(out_path):
        mkdir(out_path)

    files = sorted(listdir(folder))[:nb]
    if files == sorted(listdir(out_path))[:nb] and not force:
        print("Files already exist and preprocessed..")
        return out_path
    for fn in tqdm(files):
        with open(folder + fn, 'r', encoding="utf8") as f,\
                open(out_path + fn, "w") as fn_preprecessed:
            corpus = f.read().lower()
            for tok in ['"', '-', '--', '#', 'www', 'http']:
                corpus = corpus.replace(tok,"")
            # corpus = clean(corpus, fix_unicode=False, to_ascii=False,
            #                lower=True,
            #                no_numbers=False, replace_with_number="__NUM__",
            #                no_urls=False, replace_with_url="__URL__",
            #                no_emails=False, replace_with_email="__EMAIL__")
            fn_preprecessed.write(corpus)
    print(f" {nb} files saved at the folder: {out_path}")
    return out_path



def test_training(model_file, folder_test=folder_test):
    model = kenlm.Model(model_file)
    perplex = []
    i=0
    files = [file for file in listdir(folder_test) if path.splitext(file)[1]=='.ref']
    print(files)
    # files = listdir(folder_test)
    test_file = files[0]
    with open(folder_test+test_file, 'r',encoding="utf8") as f:
        while i < 1000:
            line = f.readline().strip()
            # perplex.append(model.score(line, bos = True, eos = True))
            if 5<=len(line.split())<=25:
                perplex.append(model.perplexity(line))
                i+=1
    f.close()
    return perplex

def train(model_file, folder,order=2,nb=9,test=False,**params):
    model_file_name = f'{model_file}{nb}.arpa'
    model_binary_name = f'{model_file}{nb}.bin'
    exist = path.exists(model_file_name)
    print(f"Model {model_binary_name}\tfound : {exist}, force retrain: {params['retrain']}")

    if not exist or params['retrain']:
        print('Training ...')
        all_files = listdir(folder)
        test_files = [file for file in all_files if path.splitext(file)[1] == '.test'][:nb]
        ref_files = [file for file in all_files if path.splitext(file)[1] == '.ref'][:nb]
        if len(ref_files) == 0:
            print("No '.ref' files found, considering all files in the folder for training..")
            ref_files = sorted(all_files[:nb])
        times = []
        sizes = []
        perplexities = []
        training_sample = "models/training_sample.txt"
        myfile = open(training_sample, "w")
        myfile.close()
        for i,fn in enumerate(ref_files):

            # comcatenate the first {0..i} text files and use them all for training
            # this would help check the impact of training data size
            system("cat "+folder+fn+ " >> "+ training_sample)
            print(f"Files 1..{i+1}:")
            #print(fns)

        start_time = time.time()
        cmd = kenlm_dir+'build/bin/lmplz -o '+str(order)+' -S 80% -T /tmp < '+training_sample+' > '+model_file_name
        #print(cmd, end='\n\n')
        system(cmd)
        t = time.time() - start_time
        system(f"{kenlm_dir}build/bin/build_binary {model_file_name} {model_binary_name}")
    else:
        t=0
    if test:
        size = path.getsize(model_binary_name) / (1024*1024) # convert to MB
        score = test_training(model_binary_name)
        print(f'Time: {t:.2f}s\t Size: {size:.2f}MB\t Score: {mean(score):.2f}')
        return t, size, score
    else:
        size = path.getsize(model_binary_name) / (1024 * 1024)
        return -1, size, -1

def load_model(model_binary_name):
    t = time.time()
    print(f"Loading model : {model_binary_name}")
    model = kenlm.Model(model_binary_name)
    loading_time = round(time.time() - t,2)
    print(f"Loading time: {loading_time:.2f}s")
    return model,loading_time

if __name__ == '__main__':
    args = get_args()
    params = vars(args)
    print(params)
    print('heads:', list(params['head']), 'backoff:', params['backoff'])

    if params['model_file'].count('/'):
        if params['folder'] == 'full':
            folder = folder_full
        elif params['folder'] == 'short':
            folder = folder_short
        else:
            folder = preprocess_folder(folder_original, args.nb,args.reprocess)
        params['folder']= folder

        model_file_name = f'{args.model_file}{args.nb}.arpa'
        model_binary_name = f'{args.model_file}{args.nb}.bin'


        model_base_name = path.splitext(model_binary_name.split('/')[1])[0]
        print(' #######################', params['scoring'])

        # heads = [int(x)  if len(params['head'])>0 else 0 for x in params['head']]




        training_time, size, perplexities = train(**params)


        # size = path.getsize(model_file_name) / (1024 * 1024)  # convert to MB
        # score = test_training(model_file_name)
        # print(f'\t Size: {size:.2f}MB\t Score: {mean(score):.2f}')

        ## Loading model takes time so we do it once in the begining
        # model_binary_name = 'models/trigram_p9.bin'
        model,loading_time = load_model(model_binary_name)
    elif  params['model_file'].lower() == 'gpt2':
        model = GPT2()
        model_base_name = 'gpt2'
        if params['scoring'] != 'perplexity':
            print('GPT 2 uses perplexity for scoring, switching to perplexity')
            params['scoring'] = 'perplexity'
    else:
        exit('Unknown model file')


    test_files = ['dev_data/news.test', 'dev_data/hans.test', 'dev_data/euro.test']

    if path.exists('results/results.json'):
        with open('results/results.json', 'r') as f:
            scores = json.load(f)
    else:
        with open('results/results.json', 'w') as f:
            pass
        scores = {}

    ref = "why does everything have to become such a big issue ?"
    sent_1 = '? everything big why to become does have such issue a'
    sent_2 = "a big issue to have become such ? why does everything"
    sent_3 = "why does everything have to become such a big ? issue"
    sent_4 = "? why does everything have to become such a big issue"


    # sent_a = "25 . rue 374618 ) abel 33 , guynemer café 478 comptoir 00 ("
    # ref_b = "café comptoir abel , 25 rue guynemer ( 00 33 478 374618 ) ."
    # print(evaluate([ref], [ref]))
    # print(evaluate([sent_3], [ref]))
    # print(evaluate([sent_4], [ref]))
    # print(evaluate([sent_1], [ref]))
    # print(evaluate([sent_2], [ref]))
    # for i in range(1,4):
    #     # print('-p-'*20)
    #     # sent = solver_full(model, sent_1,i,scoring= 'perplexity')
    #     # print(evaluate([sent], [ref]))
    #     print(f"i={i}")
    #     print('-p0' * 20)
    #     sent = solver_acc(model, sent_1, i, scoring='perplexity')
    #     print(evaluate([sent], [ref]))
    #
    #     print('-p2' * 20)
    #     sent = solver_acc_2(model, sent_1, i, scoring='perplexity')
    #     print(evaluate([sent], [ref]))


    # score = predict(model, test_file='dev_data/news.test', solve_func=solver, head_size=4)



    for test_file in test_files:
        for b in params['backoff']:
            for i in params['head']:
                test_file_base_name = path.splitext(test_file.split('/')[1])[0]
                suffix = '-' if params['scoring']=='perplexity' else '-sc-'
                # print('--'*20)
                # model_version=f"KenLM-solver-acc2-head-{i}{suffix}{test_file_base_name}-{model_base_name}"
                # score = predict(model, test_file=test_file,solve_func=solver_acc_2,
                #                 head_size=i,version=f"p{i}",verbose=0,
                #                 evaluation_only=False,scoring=params['scoring'])
                # scores[model_version] = score

                print('-0'*20)
                model_suffix = 'GPT2' if 'gpt2' in params['model_file'] else 'KenLM'

                model_version=f"{model_suffix}-solver-acc3-head-{i}-{b}-{suffix}{test_file_base_name}-{model_base_name}"
                print('->'*5,model_version)
                t = time.time()
                score = predict(model, test_file=test_file,solve_func=solver_acc_3,
                                head_size=i,version=f"p0{i}",verbose=0,
                                evaluation_only=False,scoring=params['scoring'],backoff=b)
                inference_time = time.time() - t
                scores[model_version]= score
                # training_time, size, perplexities
                if training_time>0: scores[model_version]['training_time'] = round(training_time,2)
                if size>0: scores[model_version]['model_size'] = round(size,2)
                if perplexities>0: scores[model_version]['model_score'] = round(perplexities,2)
                # loading_time
                scores[model_version]['loading_time']= round(loading_time,2)
                scores[model_version]['inference_time']= round(inference_time,2)


                with open('results/tmp.json', 'w') as f:
                    json.dump(scores, f, indent=3)
                try:
                    with open('results/results.json', 'w') as f:
                        json.dump(scores, f, indent=3)
                except:
                    print('could not write to results.json')
    # print(scores)
    with open('results/results.json', 'w') as f:
        json.dump(scores, f, indent=3)

    # predict(model, test_file='dev_data/news.test',solve_func=solver(head_size=1))
    # predict(model, test_file='dev_data/news.test',solve_func=solver(head_size=2))
    # predict(model, test_file='dev_data/news.test',solve_func=solver(head_size=3))
