## Reference: http://norvig.com/spell-correct.html

###########################################################################
########################## SPELL CHECKER ##################################

import numpy as np 
import pandas as pd # data process 
import re
from collections import Counter

#df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#df_train_corrected = df_train
df_test_corrected = df_test


def words(text): return re.findall(r'\w+', text.lower())

## big.txt is a corpus of words. It is a concatenation of public domain book excerpts from Project Gutenberg and lists of most frequent words from Wiktionary and the British National Corpus
WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


import multiprocessing as mp
import pandas.util.testing as pdt


## Added this function to use for .apply on every row
i=0
def sentence_correction(row):
    global i
    print ("Current iteration is {}.".format(i))

    q1 = str(row['question1']).lower().split()
    q2 = str(row['question2']).lower().split()
    
    q1 = [correction(word) for word in q1]
    q2 = [correction(word) for word in q2]
    
    q1 = " ".join(q1)
    q2 = " ".join(q2)
    
    row['question1'] = q1
    row['question2'] = q2
    
    i += 1

    return row

def process_df(df):
    res = df.apply(sentence_correction, axis = 1)
    return res    


if __name__ == '__main__':
    p = mp.Pool(processes=4)
    split_dfs = np.array_split(df_test2,4)
    pool_results = p.map(process_df, split_dfs)
    p.close()
    p.join()

# merging parts processed by different processes
    df_test_corrected = pd.concat(pool_results, axis=0)



#df_train_corrected = df_train.apply(sentence_correction, axis = 1)
#df_test_corrected = df_test.apply(sentence_correction, axis = 1)
