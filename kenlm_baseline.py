from os import listdir, system, path
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

folder = 'training-monolingual.tokenized.shuffled/'
folder_short = 'train_data/preprocessed-short/'
folder_full = 'train_data/preprocessed-full/'

folder_test = 'train_data/heldout/'
kenlm_dir = '/home/kacem/kenlm/'
# test_file = folder_test+'news.en-00000-of-00100'
model_file = 'models/bigram'



def get_args():

    parser = argparse.ArgumentParser(description='KenLM')
    parser.add_argument("-f", '--folder', type=str, help="the folder to use",choices=['short','full'], default='short')
    parser.add_argument("-m", '--model_file', type=str, help="the model to use", default='models/bigram')
    parser.add_argument("-v", '--verbosity', type=int, help="increase output verbosity", default=0)
    parser.add_argument("-b", '--nb', type=int, help="# files read",default=2)
    parser.add_argument("-o", '--order', type=int, help="2: bigram, 3: trigram",default=2)

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
    if r==4:
        permutations = [[seq[i], seq[j], seq[k],seq[l]] for i in n for j in n for k in n for l in n if len(set([i,j,k,l]))==4]
    elif r==3:
        permutations = [[seq[i],seq[j],seq[k]] for i in n for j in n for k in n if i != j and i != k and j != k]
    elif r==2:
        permutations = [[seq[i], seq[j]] for i in n for j in n if i != j]
    else:
        permutations= [[t] for t in seq]

    return permutations


def solver(model, sentence,head_size = 1,backoff=0,scoring='perplexity',verbose=0):
    """
    Find the most probable ordering of a sequence e of tokens using a greedy search with kenlm trigram for scoring
    we assume that sequences are longer than 3 words

    :param model (kenlm model):
    :param sentence str:
    :param head_size int: number of words in the starting head of a sentence
    :return: str
    """
    tokens = sentence.split(' ')
    permutations = find_permutations(tokens,head_size)
    if verbose:
        print(f"Found {len(permutations)} permutations with subsequence length {head_size} from sequence length: {len(tokens)}")
    best_score = 10000000 #TODO -inf if score else +inf if perplex
    best_sequence = []
    t = time.time()
    for p in permutations:
        toks = tokens[:]
        for i in range(head_size-backoff):
            toks.remove(p[i])
        new_sequence = p+toks
        score = model.perplexity(" ".join(new_sequence)) # TODO
        # score = model.score(" ".join(new_sequence), bos=True, eos=True)
        # print(seq,score)
        if score < best_score: #TODO > score or < perplexity
            best_score = score
            best_sequence = new_sequence[:]
    tokens = best_sequence[:]
    i=head_size-backoff
    while i < len(tokens):
        best_score = 10000000 ##TODO -inf if score else +inf if perplex
        best_candidate = "UNK"
        the_rest = tokens[i:]
        for t,token in enumerate(the_rest):
            new_sequence = tokens[:i] + [token]+the_rest[:t]+the_rest[t+1:]
            score = model.perplexity(" ".join(new_sequence)) #TODO
            # score = model.score(" ".join(new_sequence),bos = True, eos = True)
            if score < best_score: ##TODO > score or < perplexity
                best_score = score
                best_sequence = new_sequence[:]
        i+=1
    if verbose: print(f"Prediction time: {time.time()-t:e}s")

    result = " ".join(best_sequence)
    if verbose: print(f"{len(sentence.split())} words\t{len(sentence)} caracters:\t{sentence}\n"
          f"{len(result.split())} words\t{len(result)} caracters:\t{result} ")
    return result


def solver0(model, sentence, head_size = 1, scoring='perplexity',verbose=0):
    """
    Find the most probable ordering of a sequence e of tokens using a greedy search with kenlm trigram for scoring
    we assume that sequences are longer than 3 words

    :param model (kenlm model):
    :param sentence str:
    :return: str
    """
    tokens = sentence.split()
    sequences = find_permutations(tokens,head_size)
    if verbose:
        print(f"Found {len(sequences)} permutations with subsequence length {head_size} from sequence length: {len(tokens)}")
    best_score = 10000000 #TODO -inf if score else +inf if perplex
    best_sequence = []
    t = time.time()
    for seq in sequences:
        score = model.perplexity(" ".join(seq)) # TODO
        # score = model.score(" ".join(seq), bos=True, eos=True)
        # print(seq,score)
        if score < best_score: ##TODO > score or < perplexity
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
        for token in tokens:
            score = model.perplexity(" ".join(best_sequence+[token])) # TODO
            # score = model.score(" ".join(best_sequence+[token]), bos=True, eos=True)
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



def evaluate(sentences, references):
    """Evalute two lists of strings of same lengths

    :param sentences (List[str]): predicted sentences
    :param references (List[str]): true labels
    :return: metrics (dict)
    """
    print("\nEvaluation..")
    assert len(sentences) == len(references)
    print(f"Nb of sentences : {len(sentences)}")
    # Binary evaluation
    b = 0
    for ind in range(len(sentences)):
        if sentences[ind] == references[ind]:
            b = b + 1

    #blue score
    bleu = BLEU()
    bleu_score = bleu.corpus_score(sentences, [references]) # reference should be a list of a list

    # text distance
    d_hamming = [ textdistance.hamming.distance(sentences[inx], references[inx]) for inx in range(len(sentences)) ]
    avg_hamming = mean(d_hamming)

    d_LCSS = [textdistance.lcsstr.distance(sentences[inx], references[inx]) for inx in range(len(sentences)) ]
    avg_LCSS = mean(d_LCSS)

    pos_match = [compute_pos_match(sentences[inx], references[inx]) for inx in range(len(sentences))]
    avg_pos_match = mean(pos_match)

    metrics = {
        "match" : b / len(sentences),
        "bleu_score" : bleu_score.score,
        "avg_hamming" : avg_hamming,
        "avg_LCSS" : avg_LCSS,
        "avg_pos_match": avg_pos_match
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

def predict(model,test_file,solve_func,head_size,version="",verbose=0):
    dev_file = path.splitext(test_file)[0]+str(version)+'.dev'
    ref_file = path.splitext(test_file)[0]+'.ref'
    print(test_file,dev_file,ref_file)
    strict_match = 0
    pos_match_list = []
    with open(test_file, encoding='utf-8') as test,\
            open(dev_file,'w',encoding='utf-8') as dev,\
                open(ref_file,encoding='utf-8') as ref:
        sentences = test.read().split('\n')
        for sent in tqdm(sentences):
            if sent != "":
                prediction = solve_func(model, sent, head_size,verbose=verbose)
                dev.write(prediction+'\n')
                ref_sent = ref.readline().strip().split()
                dev_sent = prediction.strip().split()
                len_sentence = len(dev_sent)
                assert len(dev_sent) == len(ref_sent)
                if ref_sent == sent:
                    strict_match += 1
                pos_match = sum([1 for i in range(len_sentence) if ref_sent[i] == dev_sent[i]]) / len_sentence
                if verbose: print(f"pos match : {pos_match}")
                pos_match_list.append(pos_match)
                dev.flush()
    score = evaluate_files(ref_file, dev_file)
    print(version,score)

    return score

def test_training(model_file, folder_test=folder_full):
    model = kenlm.Model(model_file)
    perplex = []
    i=0
    # files = [file for file in listdir(folder) if path.splitext(file)[1]=='.test']
    files = listdir(folder_test)
    test_file = files[46]
    with open(folder_test+test_file, 'r',encoding="utf8") as f:
        while i < 1000:
            line = f.readline()
            # perplex.append(model.score(line, bos = True, eos = True))
            if 5<=len(line.split())<=25:
                perplex.append(model.perplexity(line))
                i+=1
    f.close()
    return perplex

def train(model_file, folder,order=2,nb=9,**kwargs):

    all_files = listdir(folder)
    test_files = [file for file in all_files if path.splitext(file)[1] == '.test'][:nb]
    ref_files = [file for file in all_files if path.splitext(file)[1] == '.ref'][:nb]
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
    model_file_name = f'{model_file}{nb}.arpa'
    model_binary_name = f'{model_file}{nb}.bin'
    start_time = time.time()
    cmd = kenlm_dir+'build/bin/lmplz -o '+str(order)+' -S 80% -T /tmp < '+training_sample+' > '+model_file_name
    #print(cmd, end='\n\n')
    system(cmd)
    t = time.time() - start_time
    system(f"{kenlm_dir}build/bin/build_binary {model_file_name} {model_binary_name}")
    size = path.getsize(model_file_name) / (1024*1024) # convert to MB
    score = test_training(model_file_name)
    print(f'Time: {t:.2f}s\t Size: {size:.2f}MB\t Score: {mean(score):.2f}')
    return t, size, score

if __name__ == '__main__':
    args = get_args()
    params = vars(args)
    folder = folder_full if args.folder == 'full' else folder_short
    params['folder']=folder
    model_file_name = f'{args.model_file}{args.nb}.arpa'
    model_binary_name = f'{args.model_file}{args.nb}.bin'

    # training_time, size, perplexities = train(**params)


    # size = path.getsize(model_file_name) / (1024 * 1024)  # convert to MB
    # score = test_training(model_file_name)
    # print(f'\t Size: {size:.2f}MB\t Score: {mean(score):.2f}')

    ## Loading model takes time so we do it once in the begining
    t = time.time()
    print(f"Loading model : {model_binary_name}")
    model = kenlm.Model(model_binary_name)
    loading_time = time.time() - t
    print(f"Loading time: {loading_time:.2f}s")

    ref = "why does everything have to become such a big issue ?"
    sentence = '? everything big why to become does have such issue a'

    # sent_a = "25 . rue 374618 ) abel 33 , guynemer café 478 comptoir 00 ("
    # ref_b = "café comptoir abel , 25 rue guynemer ( 00 33 478 374618 ) ."
    # print(evaluate([sentence], [ref]))
    # print(evaluate([sentence], [sentence]))
    # for i in range(1,2):
    #     print('--'*20)
    #     sent = solver(model, sent_a,i)
    #     print(evaluate([sent], [ref_b]))
    #
    #     print('-0'*20)
    #     sent = solver0(model, sent_a,i)
    #     print(evaluate([sent], [ref_b]))


    # score = predict(model, test_file='dev_data/news.test', solve_func=solver, head_size=4)

    test_files = ['dev_data/news.test', 'dev_data/hans.test', 'dev_data/euro.test' ]

    if path.exists('results/results.json'):
        with open('results/results.json', 'r') as f:
            scores = json.load(f)
    else:
        with open('results/results.json', 'w') as f:
            pass
        scores = {}

    for test_file in test_files:
        for i in range(1,4):
            print('--'*20)
            model_version=f"KenLM-solver-full-head-{i}-perplexity-{path.splitext(test_file)[0]}"
            score = predict(model, test_file=test_file,solve_func=solver,head_size=i,version=f"p{i}",verbose=0)
            scores[model_version] = score

            print('-0'*20)

            model_version=f"KenLM-solver-accumulative-head-{i}-perplexity-{path.splitext(test_file)[0]}"
            score = predict(model, test_file=test_file,solve_func=solver0,head_size=i,version=f"p0{i}",verbose=0)
            scores[model_version]= score
            with open('results/tmp.json', 'w') as f:
                json.dump(scores, f, indent=3)
    print(scores)
    with open('results/results.json', 'w') as f:
        json.dump(scores, f, indent=3)

    # predict(model, test_file='dev_data/news.test',solve_func=solver(head_size=1))
    # predict(model, test_file='dev_data/news.test',solve_func=solver(head_size=2))
    # predict(model, test_file='dev_data/news.test',solve_func=solver(head_size=3))
