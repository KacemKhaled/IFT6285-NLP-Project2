import enum
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
    parser.add_argument("-f", '--folder', type=str, help="the folder to use",choices=['short','full', 'original', 'original_preprocess'], default='full')
    parser.add_argument("-s", '--scoring', type=str, help="the score to use", choices=['perplexity', 'score'],
                        default='perplexity')
    parser.add_argument("-m", '--model_file', type=str, help="the model to use", default='models/bigram')
    parser.add_argument('--solver', type=str, help="the solver to use: acc or full or lookup",choices=['acc', 'full','lookup'], default='acc')
    parser.add_argument("-j", '--json_file_results', type=str, help="json file for results", default='results/results_final.json')
    parser.add_argument("-v", '--verbose', type=int, help="increase output verbosity", default=0)
    parser.add_argument("-n", '--nb', type=int, help="# files read",default=2)
    parser.add_argument("-d", '--head_list', nargs='+',type=int, help="head size, default : nothing",default=[0])
    parser.add_argument("-b", '--backoff_list', nargs='+',type=int, help="backoff_list, default: 1",default=[0])
    parser.add_argument("-o", '--order', type=int, help="2: bigram, 3: trigram",default=2)
    parser.add_argument("-r", '--retrain', action='store_true', help="force retraining",default=False)
    parser.add_argument("-p", '--reprocess', action='store_true', help="force reprocessing",default=False)
    parser.add_argument("-t", '--brute_force_tail',type=str, choices=['0', 't','x'], help="brute force tail",default=False)
    parser.add_argument( '--test', action='store_true', help="test the language model",default=False)

    return parser.parse_args()

def find_permutations(seq,r):
    """
    Find the possible r permutations from a sequence
    :param seq (List): sequence of str
    :param r (int): length of sub-sequence: (1,2,3)
    :return permutations (List):
    """
    # permutations = []
    # n = range(len(seq))
    # if r==3:
    #     permutations = [[seq[i],seq[j],seq[k]] for i in n for j in n for k in n if i != j and i != k and j != k]
    # elif r==2:
    #     permutations = [[seq[i], seq[j]] for i in n for j in n if i != j]
    # elif r==1:
    #     permutations = [[t] for t in seq]
    # else:
    #     permutations = [list(t) for t in itertools.permutations(seq,r)]

    return [list(t) for t in itertools.permutations(seq,r)] # permutations

def solver_full_3(model, sentence, head,
                 scoring,verbose,backoff,brute_force_tail):
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
    skips = 0
    bt=0
    while tokens:
        temp = final_sequence[:]
        if len(tokens) < 8 and brute_force_tail >= 't':
            head = 7
            bt+=1
        elif len(tokens) < 10 and brute_force_tail=='x':
            head = 5
            bt+=1
        r = min(head, len(tokens))
        permutations = find_permutations(tokens,r)
        if verbose>0:
            print(f"\nFound {len(permutations)} permutations with subsequence length {r} from sequence length: {len(tokens)}")
            # print(f" {len(permutations)} permutations : {permutations} from sequence length: {len(tokens)}: {tokens}")
        best_score = 100000000
        best_sequence = []
        t = time.time()
        for seq in permutations:
            toks = tokens[:]
            for i in range(r):
                toks.remove(seq[i])
            # the_rest = tokens[r:]
            new_sequence = final_sequence[:] + seq + toks
            if scoring == 'perplexity':
                score = model.perplexity(" ".join(new_sequence))
            elif scoring == 'score':
                score = - model.score(" ".join(new_sequence), bos=False, eos=False) # not the enf of sentence
            if verbose>0: print(" ".join(temp+seq), score)
            if verbose>0: print('best_score', best_score)
            if score < best_score:
                best_score = score
                best_sequence = seq[:]
        # else:
        #     if best_score == 100000000:
        #         # best_sequence = permutations[0]
        #         best_sequence = random.choice(permutations)
        #         skips+=1
        if len(best_sequence) > backoff and len(best_sequence) < len(tokens):
            new_range = len(best_sequence)-backoff
        else:
            new_range = len(best_sequence)
        # print(new_range)
        for i in range(new_range):
            tokens.remove(best_sequence[i])
        final_sequence.extend(best_sequence[:new_range])

    # print(f"Loading time: {loading_time:.2f}s\tPrediction time: {time.time()-t:e}s")
    if verbose>0: print(f"Prediction time: {time.time() - t:e}s")
    result = " ".join(final_sequence)
    if verbose>0: print(f"{len(sentence.split())} words\t{len(sentence)} caracters:\t{sentence}\n"
          f"{len(result.split())} words\t{len(result)} caracters:\t{result} ")
    return result,bt



def solver_acc_3(model, sentence, head,
                 scoring,verbose,backoff,brute_force_tail):
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
    skips = 0
    bt=0
    while tokens:
        temp = final_sequence[:]
        if len(tokens) < 8 and brute_force_tail >= 't':
            head = 7
            bt+=1
        elif len(tokens) < 10 and brute_force_tail=='x':
            head = 5
            bt+=1
        subsequence_length = min(head, len(tokens))
        permutations = find_permutations(tokens,subsequence_length)
        if verbose>0:
            print(f"\nFound {len(permutations)} permutations with subsequence length {subsequence_length} from sequence length: {len(tokens)}")
            # print(f" {len(permutations)} permutations : {permutations} from sequence length: {len(tokens)}: {tokens}")
        best_score = 100000000
        best_sequence = []
        t = time.time()
        for seq in permutations:
            if scoring == 'perplexity':
                score = model.perplexity(" ".join(temp+seq))
            elif scoring == 'score':
                score = - model.score(" ".join(temp+seq), bos=False, eos=False) # not the enf of sentence
            if verbose>0: print(" ".join(temp+seq), score)
            if verbose>0: print('best_score', best_score)
            if score < best_score:
                best_score = score
                best_sequence = seq[:]
        # else:
        #     if best_score == 100000000:
        #         # best_sequence = permutations[0]
        #         best_sequence = random.choice(permutations)
        #         skips+=1
        if len(best_sequence) > backoff and len(best_sequence) < len(tokens):
            new_range = len(best_sequence)-backoff
        else:
            new_range = len(best_sequence)
        # print(new_range)
        for i in range(new_range):
            tokens.remove(best_sequence[i])
        final_sequence.extend(best_sequence[:new_range])

    # print(f"Loading time: {loading_time:.2f}s\tPrediction time: {time.time()-t:e}s")
    if verbose>0: print(f"Prediction time: {time.time() - t:e}s")
    result = " ".join(final_sequence)
    if verbose>0: print(f"{len(sentence.split())} words\t{len(sentence)} caracters:\t{sentence}\n"
          f"{len(result.split())} words\t{len(result)} caracters:\t{result} ")
    return result,bt

def lookup_construct_ref(dict,folder=folder_full):
    files = [file for file in listdir(folder) if path.splitext(file)[1]=='.ref']
    # dict = {}
    for file in tqdm(files):
        with open(folder+file,encoding='utf-8') as ref:
            for sent in ref.read().split('\n')[:-1]:
                dict[" ".join(sorted(sent.split(' ')))] = sent
    return dict
            
def solver_lookup(sentence,dict):
    sorted_sent = " ".join(sorted(sentence.split(' ')))
    if sorted_sent in dict:
        return dict[sorted_sent], 1
    else:
        return sentence, 0


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
        print(sent_ref[:2])
        print(sent_ref[-1])
        print(sent_dev[:2])
        print(sent_dev[-1])

        len_sentence = len(sent_ref)
        assert len(sent_ref) == len(sent_dev)
        metrics  = evaluate(sent_dev, sent_ref)


    return metrics


def predict_lookup(test_file,version="",
            evaluation_only=False,verbose=0,dict={}):
    dev_file = path.splitext(test_file)[0]+str(version)+'.dev'
    ref_file = path.splitext(test_file)[0]+'.ref'
    print(test_file,dev_file,ref_file)
    all_found = 0
    if not evaluation_only:
        with open(test_file, encoding='utf-8') as test,\
                open(dev_file,'w',encoding='utf-8') as dev:
            sentences = test.read().split('\n')
            for sent in tqdm(sentences):
                if sent != "":
                    prediction, found = solver_lookup(sent,dict)
                    all_found+=found
                    if verbose<0:
                        if found: print(f"found {found} sentence: {sent}")
                    dev.write(prediction+'\n')
                    dev.flush()
    score = evaluate_files(ref_file, dev_file)
    print(version,score)
    print(f"all_found: {all_found}")
    return score

def predict(model,test_file,solve_func,version="",
            evaluation_only=False,head=1,scoring='perplexity',verbose=0,backoff=0,brute_force_tail='0'):
    dev_file = path.splitext(test_file)[0]+str(version)+'.dev'
    ref_file = path.splitext(test_file)[0]+'.ref'
    print(test_file,dev_file,ref_file)
    all_skips = []
    if not evaluation_only:
        with open(test_file, encoding='utf-8') as test,\
                open(dev_file,'w',encoding='utf-8') as dev:
            sentences = test.read().split('\n')
            for sent in tqdm(sentences):
                if sent != "":
                    prediction, skips = solve_func(model, sent,
                                                   head,scoring,verbose=verbose,backoff=backoff,
                                                   brute_force_tail=brute_force_tail)
                    if skips: all_skips.append(skips)
                    if verbose<0:
                        if skips: print(f"skipped {skips} choices")
                    dev.write(prediction+'\n')
                    dev.flush()
    score = evaluate_files(ref_file, dev_file)
    print(version,score)
    print(f"all_skips: {len(all_skips)},Total: {sum(all_skips)}")
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
        t= time.time()
        while i < 1000:
            line = f.readline().strip()
            # perplex.append(model.score(line, bos = True, eos = True))
            if 5<=len(line.split())<=25:
                perplex.append(model.perplexity(line))
                i+=1
    
    return mean(perplex),(time.time()-t)/1000

def train(model_file, folder,order=2,nb=9,test=False,**params):
    model_file_name = f'{model_file}-{nb}-{folder[-4:-1]}.arpa'
    model_binary_name = f'{model_file}-{nb}-{folder[-4:-1]}.bin'
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
        score,qr_time = test_training(model_binary_name)
        print(f'Time: {t:.2f}s\t Size: {size:.2f}MB\t Score: {score:.2f}, Query time: {qr_time:.2f}')
        return t, size, score, qr_time
    else:
        size = path.getsize(model_binary_name) / (1024 * 1024)
        return -1, size, -1, -1
def load_or_create_json(file):
    if path.exists(file):
        with open(file, 'r') as f:
            scores = json.load(f)
    else:
        with open(file, 'w') as f:
            pass
        scores = {}
    return scores

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
    print('heads:', list(params['head_list']), 'backoff_list:', params['backoff_list'])

    if params['model_file'].count('/'):
        if params['folder'] == 'full':
            folder = folder_full
        elif params['folder'] == 'short':
            folder = folder_short
        elif params['folder'] == 'original':
            folder = folder_original
        else:
            folder = preprocess_folder(folder_original, args.nb,args.reprocess)
            print(folder)
        params['folder']= folder

        model_file_name = f'{params["model_file"]}-{params["nb"]}-{params["folder"][-4:-1]}.arpa'
        model_binary_name = f'{params["model_file"]}-{params["nb"]}-{params["folder"][-4:-1]}.bin'


        model_base_name = path.splitext(model_binary_name.split('/')[1])[0]
        print(' #######################', params['scoring'])

        # heads = [int(x)  if len(params['head_list'])>0 else 0 for x in params['head_list']]




        training_time, size, perplexities, qr_time = train(**params)


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
        print('Unknown model file')
        model_base_name = 'lookup'
        params['scoring'] = 'find'


    test_files = ['dev_data/news.test', 'dev_data/hans.test', 'dev_data/euro.test','dev_data/projet2-fic.txt']
    results_file = params['json_file_results']
    # if path.exists(results_file):
    #     with open(results_file, 'r') as f:
    #         scores = json.load(f)
    # else:
    #     with open(results_file, 'w') as f:
    #         pass
    #     scores = {}
    scores = load_or_create_json(results_file)
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


    # score = predict(model, test_file='dev_data/news.test', solve_func=solver, head=4)

    if 'lookup' in params['solver']:
        # dict_file_name = 'results/dict.json'
        # dict_sent = load_or_create_json(dict_file_name)
        # if len(dict_sent) ==0:
        dict_sent = {}
        dict_sent = lookup_construct_ref(dict_sent)
    for test_file in test_files[:3]:
        for b in params['backoff_list']:
            for i in params['head_list']:
                if b<i:
                    test_file_base_name = path.splitext(test_file.split('/')[1])[0]
                    suffix = '-pr-' if params['scoring']=='perplexity' else '-sc-'
                    suffix = params['brute_force_tail'] + suffix
                    # print('--'*20)
                    # model_version=f"KenLM-acc2-head-{i}{suffix}{test_file_base_name}-{model_base_name}"
                    # score = predict(model, test_file=test_file,solve_func=solver_acc_2,
                    #                 head=i,version=f"p{i}",verbose=0,
                    #                 evaluation_only=False,scoring=params['scoring'])
                    # scores[model_version] = score
    
                    print('-0'*20)
                    model_suffix = 'GPT2' if 'gpt2' in params['model_file'] else 'KenLM'
    
                    model_version=f"{model_suffix}-{params['solver']}3-head-{i}-{b}-{suffix}{test_file_base_name}-{model_base_name}"
                    print('->'*5,model_version)
                    t = time.time()
                    # predict(model, test_file, solve_func, version="",
                    #         evaluation_only=False, head=1, scoring='perplexity', verbose=0, backoff=0)
                    if 'acc' in params['solver']:
                        solve_func = solver_acc_3
                    elif 'full' in params['solver']:
                        solve_func = solver_full_3
                    elif 'lookup' in params['solver']:
                        
                        solve_func = solver_lookup
                        score = predict_lookup( test_file=test_file,
                                    evaluation_only=False,
                                    version=f"{model_version}",verbose=params['verbose'])
                    else:
                        s = params['solver']
                        print(f'Undefined solver function: {s}')
                        exit()
                    if params['solver']!='lookup':
                        score = predict(model, test_file=test_file,
                                        solve_func=solve_func,
                                        evaluation_only=False,
                                        version=f"{model_version}",
                                        head=i,verbose=params['verbose'],
                                        scoring=params['scoring'],backoff=b,
                                        brute_force_tail=params['brute_force_tail'])
                    execution_time = time.time() - t
                    scores[model_version]= score
                    # training_time, size, perplexities
                    if params['model_file'].count('/'):
                        if training_time>0: scores[model_version]['training_time'] = round(training_time,2)
                        if size>0: scores[model_version]['model_size'] = round(size,2)
                        if perplexities>0: scores[model_version]['model_score'] = round(perplexities,2)
                        if qr_time>0: scores[model_version]['qr_time'] = round(qr_time,2)
                        # loading_time
                        scores[model_version]['loading_time']= round(loading_time,2)
                    scores[model_version]['execution_time']= round(execution_time,2)
    
    
                    with open('results/tmp.json', 'w') as f:
                        json.dump(scores, f, indent=3)
                    try:
                        with open(results_file, 'w') as f:
                            json.dump(scores, f, indent=3)
                    except:
                        print(f'could not write to {results_file}')
    # print(scores)
    with open(results_file, 'w') as f:
        json.dump(scores, f, indent=3)

    # predict(model, test_file='dev_data/news.test',solve_func=solver(head=1))
    # predict(model, test_file='dev_data/news.test',solve_func=solver(head=2))
    # predict(model, test_file='dev_data/news.test',solve_func=solver(head=3))
