
################################################################################
######################### 1. LIBRARIES AND DATA LOADING ########################
################################################################################

#### 1.1) Import the libraries used ###

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from nltk.corpus import stopwords 
from nltk.metrics import jaccard_distance
from collections import Counter 
from fuzzywuzzy import fuzz

from sklearn.cross_validation import train_test_split 
import xgboost as xgb 

random = 12357
np.random.seed(random)

#### 1.2) Read the data ###

# For reading in the spell corrected df_train file
df_train = pd.read_csv('train_corrected.csv', encoding = "ISO-8859-1")
df_test = pd.read_csv('test_corrected.csv',encoding = "ISO-8859-1")

################################################################################
############################## 2. DEFINE FUNCTIONS #############################
################################################################################

### 2.1) Set 1: Shared words (5 features)  ###

stops = set(stopwords.words("english"))
    
def shared_words(row):
    

    q1words = {}
    q2words = {}
    q1 = str(row['question1']).lower().split()
    q2 = str(row['question2']).lower().split()
    
    for word in str(row['question1']).lower().split(): 
        if word not in stops: 
            q1words[word] = 1 

    for word in str(row['question2']).lower().split(): 
        if word not in stops:
            q2words[word] = 1
    
    if len(q1words) == 0 or len(q2words) == 0 :
        return '0:0:0:0:0:0:0'
    
    # Common non stop words between question pairs. Both variables are equivalent 
    shared_words_q1 = [word for word in q1words.keys() if word in q2words.keys()]
    shared_words_q2 = [word for word in q2words.keys() if word in q1words.keys()]   
    
    
    
    R = (len(shared_words_q1) + len(shared_words_q2))/(len(q1words) + len(q2words))
    R1 = len(q1words) / len(q1) # q1 non stop words ratio
    R2 = len(q2words) / len(q2)
    R3 = R1-R2
    R4 = len(shared_words_q1)
    
    hammer = sum(1 for i in zip(q1, q2) if i[0]==i[1])/max(len(q1), len(q2))
    
    #shared 2 gram 
    q1_2gram = set([i for i in zip(q1, q1[1:])])
    q2_2gram = set([i for i in zip(q2, q2[1:])])
    shared_2gram = q1_2gram.intersection(q2_2gram)
    
    
    if len(q1_2gram) + len(q2_2gram) == 0:
        R2gram = 0
    else:
        R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))
    
    
    return '{}:{}:{}:{}:{}:{}:{}'.format(R,R1,R2,R3,R4,hammer,R2gram)
    

### 2.2) Set 2: TDIDF (4 features)  ###

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist() + df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
words = (" ".join(train_qs)).lower().split() 
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()} 


def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    q1 = str(row['question1']).lower().split()
    q2 = str(row['question2']).lower().split()
    
    
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return '0:0:0'
    

    shared_weights = [weights.get(word,0) for word in q1words if word in q2words] + [weights.get(word,0) for word in q2words if word in q1words]   
    total_weights = [weights.get(word,0) for word in q1words] + [weights.get(word,0) for word in q2words]
    R = np.sum(shared_weights) / np.sum(total_weights)
    
    
    q1_weights = [weights.get(word,0) for word in q1words]
    q1_total_weights = [weights.get(word,0) for word in q1]
    R1 = np.sum(q1_weights) / np.sum(q1_total_weights)
    
    q2_weights = [weights.get(word,0) for word in q2words]
    q2_total_weights = [weights.get(word,0) for word in q2]
    R2 = np.sum(q2_weights) / np.sum(q2_total_weights)
    
    return '{}:{}:{}'.format(R,R1,R2)

### 2.3) Set 5: Distances (Jaccard, Cosine, Euclidean, Manhattan) (2 features) ###
# Side note: Euclidean and manhattan currently not included in final script
    
    
def jaccard_dist(row):
    return jaccard_distance(set(str(row['question1'])), set(str(row['question2'])))
    

def cosine_dist(row):
    a = set(str(row['question1']))
    b = set(str(row['question2']))
    d = len(a)*len(b)
    if (d == 0):
        return 0
    else: 
        return len(a.intersection(b))/d
    

def euclidean_dist(row):
    a = len(set(str(row['question1'])))
    b = len(set(str(row['question2'])))
    d = np.sqrt(np.square(a-b))
    if (d == 0):
        return 0
    else:
        return d 
    
    
def manhattan_dist(row):
    a = len(set(str(row['question1'])))
    b = len(set(str(row['question2'])))
    d = np.absolute(a-b)
    if (d == 0):
        return 0
    else:
        return d 



import jellyfish

def jelly_features(row):
    q1 = str(row['question1'])
    q2= str(row['question2'])
    
    LD = jellyfish.levenshtein_distance(q1,q2)
    DLD = jellyfish.damerau_levenshtein_distance(q1,q2)
    JD = jellyfish.jaro_distance(q1,q2)
    JW = jellyfish.jaro_winkler(q1,q2)
    
    return '{}:{}:{}:{}'.format(LD,DLD,JD,JW)



    
################################################################################
############################ 3. FEATURES ENGINEERING ###########################
################################################################################


# Currently 60 features
    # Set 1 (7 features)
    # 1.1 = Proportion of shared words
    # 1.2 = Ratio of q1's non stopwords
    # 1.3 = Ratio of q2's non stopwords
    # 1.4 = Ratio difference (1.3 - 1.4)
    # 1.5 = Length (number) of shared non stop words
    # 1.6 = Hammering distance
    # 1.7 = Shared 2 gram

    
    #Set 2 (4 features)
    # 2.1 = TFIDF of shared words between question pairs
    # 2.2 = TFIDF of q1 
    # 2.3 = TFIDF of q2
    # 2.4 = TFIDF difference between 2.2 - 2.3
    
    # Set 3 (12 features)
    # 3.1 = Word count in q1
    # 3.2 = Word count in q2
    # 3.3 = Word count difference 
    # 3.4 = Character count in q1 (including spaces)
    # 3.5 = Character count in q2 (including spaces)
    # 3.6 = Character count difference (3.4 - 3.5 )
    # 3.7 = Character count in q1 (no spaces)
    # 3.8 = Character count in q2 (no spaces)
    # 3.9 = Character count differences (3.6-3.7)
    # 3.10 = Character per word Q1 
    # 3.11 = Character per word Q2 
    # 3.12 = Difference between 1.6 and 1.7
    
    # Set 4 (7 features - FuzzyWuzzy)
    # 4.1 = QRatio
    # 4.2 = WRatio
    # 4.3 = Partial ratio
    # 4.4 = Partial token set ratio
    # 4.5 = Partial token sort ratio
    # 4.6 = Token set ratio
    # 4.7 = Token sort ratio
    
    # Set 5 (4 features) (Potential to add more under this set!)
    #
    # Basic distance features on questions' length  
    # 5.1 = Jaccard distance
    # 5.2 = Cosine distance
    # 5.3 = Euclidean distance 
    # 5.4 = Manhattan distance 
    
    # Set Jelly (4 features) 
    # Distance features  
    #
    # 1 = LD = Levenshtein Distance
    # 2 = DLD = Damerau-Levenshtein Distance
    # 3 = JD = Jaro Distance
    # 4 = JW = Jaro-Winkler Distance
    
    
    # Set 6 (6 LSA features) - because of the complexity, i will separate set 6 
    # Distance features based on LSA-TFIDF components 
    #
    # 6.1 = Euclidean distance on LSA
    # 6.2 = Manhattan distance on LSA
    # 6.3 = Q1 component 1 and 2 (2 features)
    # 6.4 = Q2 compinent 1 and 2 (2 features)
    
    
    # Set 7 (3 'Magic' features) 
    #
    # 7.1 = Hash1 (Dropped) 
    # 7.2 = Hash2 (Dropped)
    # 7.3 = Freq of Hash1
    # 7.4 = Freq of Hash2
    # 7.5 = Intersection of q1 and q2
    
    # Set 8 (Abhishek's 13 features)
    #
    # 8.1 = Word mover distance
    # 8.2 = Normalized word mover distance
    # 8.3 = Cosine distance btwn vectors of q1 and q2
    # 8.4 = Manhattan distance btwn vectors of q1 and q2
    # 8.5 = Jaccard distance btwn vectors of q1 and q2
    # 8.6 = Canberra distance btwn vectors of q1 and q2
    # 8.7 = Euclidean distance btwn vectors of q1 and q2
    # 8.8 = Minkowski distance btwn vectors of q1 and q2
    # 8.9 = Braycurtis distance btwn vectors of q1 and q2
    # 8.10 = Skew of q1 vector
    # 8.11 = Skew of q2 vector
    # 8.12 = Kurtosis of q1 vector
    # 8.13 = Kurtosis of q2 vector
    
    # Set 9 (Locations features)
    #
    # 9.1 = No. of countries detected in q1
    # 9.2 = No. of countries detected in q2
    # 9.3 = No. of common countries btwn q1 and q2
    # 9.4 = No. of non-matched countries btwn q1 and q2
    
    
    
### 3.1) Creation of dataframes for training and testing  ###

x_train = pd.DataFrame() # hold the training set 
x_test = pd.DataFrame() # hold the testing set 
temp_df = pd.DataFrame() # to be removed later
temp_df_test = pd.DataFrame()

### 3.2) Generating and loading the features  ###
    
################################## TRAINING SET ################################

# create temp df for set 1 function  
temp_df['allR'] = df_train.apply(shared_words, axis = 1, raw = True)
temp_df['tfidf_all'] = df_train.apply(tfidf_word_match_share, axis = 1, raw = True)
temp_df['jelly'] = df_train.apply(jelly_features, axis = 1, raw = True)
# Set 1
x_train['shared_words'] = temp_df['allR'].apply(lambda x: float(x.split(':')[0]))
x_train['q1_ns_ratio'] = temp_df['allR'].apply(lambda x: float(x.split(':')[1]))
x_train['q2_ns_ratio'] = temp_df['allR'].apply(lambda x: float(x.split(':')[2]))
x_train['ratio_diff'] = temp_df['allR'].apply(lambda x: float(x.split(':')[3]))
x_train['shared_words_length'] = temp_df['allR'].apply(lambda x: float(x.split(':')[4]))
x_train['hammering'] = temp_df['allR'].apply(lambda x: float(x.split(':')[5]))
x_train['shared_2gram'] = temp_df['allR'].apply(lambda x: float(x.split(':')[6]))

# Set 2
x_train['tfidf'] = temp_df['tfidf_all'].apply(lambda x: float(x.split(':')[0]))
x_train['tfidf_q1'] = temp_df['tfidf_all'].apply(lambda x: float(x.split(':')[1]))
x_train['tfidf_q2'] = temp_df['tfidf_all'].apply(lambda x: float(x.split(':')[2]))
x_train['tfidf_diff'] = x_train['tfidf_q1'] - x_train['tfidf_q2'] 

# Set 3
x_train['q1_word_count'] = df_train['question1'].apply(lambda x: len(str(x).lower().split()))
x_train['q2_word_count'] = df_train['question2'].apply(lambda x: len(str(x).lower().split()))
x_train['diff_word_count'] = x_train['q1_word_count'] - x_train['q2_word_count']

x_train['q1_char_count_withspace'] = df_train['question1'].apply(lambda x: len(str(x)))
x_train['q2_char_count_withspace'] = df_train['question2'].apply(lambda x: len(str(x)))
x_train['diff_char_count_withspace'] = x_train['q1_char_count_withspace'] - x_train['q2_char_count_withspace']

x_train['q1_char_count_nospace'] = df_train['question1'].apply(lambda x: len(str(x).replace(' ','')))
x_train['q2_char_count_nospace'] = df_train['question2'].apply(lambda x: len(str(x).replace(' ','')))
x_train['diff_char_count_nospace'] = x_train['q1_char_count_nospace'] - x_train['q2_char_count_nospace'] 

x_train['char_per_word_q1'] = x_train['q1_char_count_nospace'] / x_train['q1_word_count']
x_train['char_per_word_q2'] = x_train['q2_char_count_nospace'] / x_train['q2_word_count']
x_train['diff_char_per_word'] = x_train['char_per_word_q1'] - x_train['char_per_word_q2']

# Set 4
x_train['fuzz_qratio'] = df_train.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_WRatio'] = df_train.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_partial_ratio'] = df_train.apply(lambda x: fuzz.partial_ratio(str(x['question1']).lower(), str(x['question2']).lower()), axis=1)
x_train['fuzz_partial_token_set_ratio'] = df_train.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_partial_token_sort_ratio'] = df_train.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_token_set_ratio'] = df_train.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_token_sort_ratio'] = df_train.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

# Set 5
x_train['jaccard_dist'] = df_train.apply(jaccard_dist, axis = 1)
x_train['cosine_dist'] = df_train.apply(cosine_dist, axis = 1)
x_train['euclidean_dist'] = df_train.apply(euclidean_dist, axis = 1)
x_train['manhattan_dist'] = df_train.apply(manhattan_dist, axis = 1)


# set jelly

x_train['LD'] = temp_df['jelly'].apply(lambda x: float(x.split(':')[0]))
x_train['DLD'] = temp_df['jelly'].apply(lambda x: float(x.split(':')[1]))
x_train['JD'] = temp_df['jelly'].apply(lambda x: float(x.split(':')[2]))
x_train['JW'] = temp_df['jelly'].apply(lambda x: float(x.split(':')[3]))



del temp_df

################################################################################
################################################################################

################################## TESTING SET ################################

# create temp df for set 1 function
temp_df_test['allR'] = df_test.apply(shared_words, axis = 1, raw = True)
temp_df_test['tfidf_all'] = df_test.apply(tfidf_word_match_share, axis = 1, raw = True)
temp_df_test['jelly'] = df_test.apply(jelly_features, axis = 1, raw = True)


# Set 1 
x_test['shared_words'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[0]))
x_test['q1_ns_ratio'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[1]))
x_test['q2_ns_ratio'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[2]))
x_test['ratio_diff'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[3]))
x_test['shared_words_length'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[4]))
x_test['hammering'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[5]))
x_test['shared_2gram'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[6]))

# Set 2
x_test['tfidf'] = temp_df_test['tfidf_all'].apply(lambda x: float(x.split(':')[0]))
x_test['tfidf_q1'] = temp_df_test['tfidf_all'].apply(lambda x: float(x.split(':')[1]))
x_test['tfidf_q2'] = temp_df_test['tfidf_all'].apply(lambda x: float(x.split(':')[2]))
x_test['tfidf_diff'] = x_test['tfidf_q1'] - x_test['tfidf_q2'] 


# Set 3
x_test['q1_word_count'] = df_test['question1'].apply(lambda x: len(str(x).lower().split()))
x_test['q2_word_count'] = df_test['question2'].apply(lambda x: len(str(x).lower().split()))
x_test['diff_word_count'] = x_test['q1_word_count'] - x_test['q2_word_count']

x_test['q1_char_count_withspace'] = df_test['question1'].apply(lambda x: len(str(x)))
x_test['q2_char_count_withspace'] = df_test['question2'].apply(lambda x: len(str(x)))
x_test['diff_char_count_withspace'] = x_test['q1_char_count_withspace'] - x_test['q2_char_count_withspace']


x_test['q1_char_count_nospace'] = df_test['question1'].apply(lambda x: len(str(x).replace(' ','')))
x_test['q2_char_count_nospace'] = df_test['question2'].apply(lambda x: len(str(x).replace(' ','')))
x_test['diff_char_count_nospace'] = x_test['q1_char_count_nospace'] - x_test['q2_char_count_nospace'] 

x_test['char_per_word_q1'] = x_test['q1_char_count_nospace'] / x_test['q1_word_count']
x_test['char_per_word_q2'] = x_test['q2_char_count_nospace'] / x_test['q2_word_count']
x_test['diff_char_per_word'] = x_test['char_per_word_q1'] - x_test['char_per_word_q2']

# Set 4
x_test['fuzz_qratio'] = df_test.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_WRatio'] = df_test.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_partial_ratio'] = df_test.apply(lambda x: fuzz.partial_ratio(str(x['question1']).lower(), str(x['question2']).lower()), axis=1)
x_test['fuzz_partial_token_set_ratio'] = df_test.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_partial_token_sort_ratio'] = df_test.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_token_set_ratio'] = df_test.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_token_sort_ratio'] = df_test.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

# Set 5
x_test['jaccard_dist'] = df_test.apply(jaccard_dist, axis = 1)
x_test['cosine_dist'] = df_test.apply(cosine_dist, axis = 1)
x_test['euclidean_dist'] = df_test.apply(euclidean_dist, axis = 1)
x_test['manhattan_dist'] = df_test.apply(manhattan_dist, axis = 1)

# set jelly 
x_test['LD'] = temp_df_test['jelly'].apply(lambda x: float(x.split(':')[0]))
x_test['DLD'] = temp_df_test['jelly'].apply(lambda x: float(x.split(':')[1]))
x_test['JD'] = temp_df_test['jelly'].apply(lambda x: float(x.split(':')[2]))
x_test['JW'] = temp_df_test['jelly'].apply(lambda x: float(x.split(':')[3]))


# remove temp 
del temp_df_test 



################################################################################
################################ LSA components  ###############################
################################################################################

import sklearn
# Import all of the scikit learn stuff
from __future__ import print_function
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans


# https://math.stackexchange.com/questions/139600/how-to-calculate-the-euclidean-and-manhattan-distance
def euclidean_distance(row):
    a1 = row['q1_c1']
    b1 = row['q1_c2']
    
    a2 = row['q2_c1']
    b2 = row['q2_c2']
    
    v = np.sqrt( np.square(a1 - a2) + np.square(b1 - b2) )
    return v 


def manhattan_distance(row):
    a1 = row['q1_c1']
    b1 = row['q1_c2']
    
    a2 = row['q2_c1']
    b2 = row['q2_c2']

    v = np.absolute(a1 - a2) + np.absolute(b1 -b2)
    return v 

    
# make the entire training set as 1 document. 
# transform q1 and q2 to linear combinations of LSA 

# recall, index 1-404290 is q1, 404291 to 808582
all_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist() + df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

all_qs = all_qs.tolist()
vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english')
dtm = vectorizer.fit_transform(all_qs) 



lsa = TruncatedSVD(2, algorithm = 'randomized')
dtm_lsa = lsa.fit_transform(dtm)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

# lets view 
pd.DataFrame(dtm_lsa, index = all_qs, columns = ["component_1","component_2"])

# convert to DF
temp_component = pd.DataFrame(dtm_lsa, columns = ["component_1","component_2"])
  
# now we need to split them 
temp_component_train_q1 = temp_component[:404290]
temp_component_train_q2 = temp_component[404290:808580]
temp_component_train_q2.index = range(404290)


temp_component_test_q1 = temp_component[808580:3154376]
temp_component_test_q2 = temp_component[3154376:5500172]
temp_component_test_q1.index = range(2345796)
temp_component_test_q2.index = range(2345796)



# recombine them 
temp_train_vector = pd.DataFrame()
temp_train_vector['q1_c1'] = temp_component_train_q1['component_1']
temp_train_vector['q1_c2'] = temp_component_train_q1['component_2']
temp_train_vector['q2_c1'] = temp_component_train_q2['component_1']
temp_train_vector['q2_c2'] = temp_component_train_q2['component_2']

# remove 
del temp_component_train_q1, temp_component_train_q2


# do for test set 

temp_test_vector = pd.DataFrame()
temp_test_vector['q1_c1'] = temp_component_test_q1['component_1']
temp_test_vector['q1_c2'] = temp_component_test_q1['component_2']
temp_test_vector['q2_c1'] = temp_component_test_q2['component_1']
temp_test_vector['q2_c2'] = temp_component_test_q2['component_2']

del temp_component_test_q1, temp_component_test_q2



#######################################
# lets do some distancing features  
#######################################


distances = pd.DataFrame()
distances['euclidean'] = temp_train_vector.apply(euclidean_distance, axis = 1)
distances['manhattan'] = temp_train_vector.apply(manhattan_distance, axis = 1)



# for the test
distances_test = pd.DataFrame()
distances_test['euclidean'] = temp_test_vector.apply(euclidean_distance, axis = 1)
distances_test['manhattan'] = temp_test_vector.apply(manhattan_distance, axis = 1)


# add back to x_train and x_test as features 


# 6 new features 
x_train['euclidean'] = distances['euclidean']
x_train['manhattan'] = distances['manhattan']

x_train['q1_c1'] = temp_train_vector['q1_c1'] 
x_train['q1_c2'] = temp_train_vector['q1_c2'] 
x_train['q2_c1'] = temp_train_vector['q2_c1'] 
x_train['q2_c2'] = temp_train_vector['q2_c2']


x_test['euclidean'] = distances_test['euclidean']
x_test['manhattan'] = distances_test['manhattan']

x_test['q1_c1'] = temp_test_vector['q1_c1'] 
x_test['q1_c2'] = temp_test_vector['q1_c2'] 
x_test['q2_c1'] = temp_test_vector['q2_c1'] 
x_test['q2_c2'] = temp_test_vector['q2_c2']



################################################################################
############################### Magic Features 1 ###############################
################################################################################

# Create duplicates of the df_train and df_test sets
df1 = df_train[['question1']].copy()
df2 = df_train[['question2']].copy()
df1_test = df_test[['question1']].copy()
df2_test = df_test[['question2']].copy()

# Rename the question2 column to question1
df2.rename(columns = {'question2':'question1'},inplace=True)
df2_test.rename(columns = {'question2':'question1'},inplace=True)

# Append all the questions to 1 single column
train_questions = df1.append(df2)
train_questions = train_questions.append(df1_test)
train_questions = train_questions.append(df2_test)

train_questions.drop_duplicates(subset = ['question1'],inplace=True)

train_questions.reset_index(inplace=True,drop=True)

# Create a dictionary, key = question, value = an unique 'ID' assigned to the question
questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()

# Create duplicates of the df_train and df_test sets
train_cp = df_train.copy()
test_cp = df_test.copy()

# Drop the qid1 and qid2 columns
train_cp.drop(['qid1','qid2'],axis=1,inplace=True)

# Change the is_duplicate values to -1 so that it can be separated easily from train set
test_cp['is_duplicate'] = -1
test_cp.rename(columns={'test_id':'id'},inplace=True)

# Does a 'rbind' between the train set and test set into one single dataframe set
comb = pd.concat([train_cp,test_cp])

# Generates the 2 features - assign the unique id to the corresponding question
comb['q1_hash'] = comb['question1'].map(questions_dict)
comb['q2_hash'] = comb['question2'].map(questions_dict)

# Creates 2 dictionaries:
    # q1_vc counts the number of hash ID occurences among all question1's
    # q1_vc counts the number of hash ID occurences among all question2's
q1_vc = comb.q1_hash.value_counts().to_dict()
q2_vc = comb.q2_hash.value_counts().to_dict()

def try_apply_dict(x,dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0
    
# Map to frequency space
# q1_freq = number of occurences for question1 in q1_vc + q2_vc
# q2_freq = number of occurences for question2 in q1_vc + q2_vc
comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

# Splits up the comb dataset into train_comb and test_comb 
train_comb = comb[comb['is_duplicate'] >= 0][['id','q1_hash','q2_hash','q1_freq','q2_freq','is_duplicate']]
test_comb = comb[comb['is_duplicate'] < 0][['id','q1_hash','q2_hash','q1_freq','q2_freq']]

# Transfer the generated features into x_train and x_test
features = ['q1_hash','q2_hash','q1_freq','q2_freq']
x_train[features] = train_comb[features]
x_test[features] = test_comb[features]

# Delete the generated variables in 'Magic Features' script
del train_questions, train_cp, test_cp, comb, train_comb, test_comb, q1_vc, q2_vc, features

################################################################################
############################### Magic Features 2 ###############################
################################################################################

from collections import defaultdict

# Create copies of original df_train and df_test
train_copy = df_train.copy()
test_copy = df_test.copy()

# Concatenates (like rbind) both train and test datasets for q1 and q2 columns only
ques = pd.concat([train_copy[['question1', 'question2']], \
        test_copy[['question1', 'question2']]], axis=0).reset_index(drop='index')

# Creates the dictionary
q_dict = defaultdict(set)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])
        
# Defines the function for returning number of frequency
def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

# Generates the feature
x_train['q1_q2_intersect'] = train_copy.apply(q1_q2_intersect, axis=1, raw=True)
x_test['q1_q2_intersect'] = test_copy.apply(q1_q2_intersect, axis=1, raw=True)

stops = set(stopwords.words("english"))
def word_match_share(q1, q2, stops=None):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    q1words = {}
    q2words = {}
    for word in q1:
        if word not in stops:
            q1words[word] = 1
    for word in q2:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0.
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

# Second dictionary for 2nd magic feature
q_dict2 = defaultdict(dict)
for i in range(ques.shape[0]):
        wm = word_match_share(ques.question1[i], ques.question2[i], stops=stops)
        q_dict2[ques.question1[i]][ques.question2[i]] = wm
        q_dict2[ques.question2[i]][ques.question1[i]] = wm
        
# Function for feature        
def q1_q2_wm_ratio(row):
    q1 = q_dict2[row['question1']]
    q2 = q_dict2[row['question2']]
    inter_keys = set(q1.keys()).intersection(set(q2.keys()))
    if(len(inter_keys) == 0): return 0.
    inter_wm = 0.
    total_wm = 0.
    for q,wm in q1.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    for q,wm in q2.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    if(total_wm == 0.): return 0.
    return inter_wm/total_wm

# Generate the features for wm ratio
x_train['q1_q2_wm_ratio'] = train_copy.apply(q1_q2_wm_ratio, axis=1, raw=True)
x_test['q1_q2_wm_ratio'] = test_copy.apply(q1_q2_wm_ratio, axis=1, raw=True)

# Delete the generated variables
del q_dict, ques, train_copy, test_copy, q_dict2


################################################################################
################################ 4. TRAINING SAMPLES ###########################
################################################################################


y_train = df_train['is_duplicate']

pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]

# now we over sample the negative class (need more reference on this methodology)
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train)))

# rebild the x_train and y_train 
x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train



x_train_cv, x_valid, y_train_cv, y_valid = train_test_split(x_train, y_train, 
                                                            test_size = 0.2, random_state = random)

################################################################################
################################# 5. XGBOOST ###################################
################################################################################

# Go straight to 5a then 5c to run model for submittion

################################################################################
################################# 5.A PARAMETERS #############################
################################################################################

### 5.1) Setting the model parameters  ###
params = {} # dict 
params['eta'] = 0.1
params['max_depth'] = 5 
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['seed'] = random


################################################################################
################################# 5.B CROSS-VALIDATE ###########################
################################################################################

# WARNING: 
# DO NOT RUN THIS ENITRE SECTION IF WE WANT TO MODEL TO SUBMIT. THIS WILL 
# ONLY CREATE A MODEL CONSISTING OF 80% OF TRAIN SET     
    

# USE THIS STEP TO OBTAIN IDEAL NROUNDS AND CHECK OVERFITTING

### 5.2) Concatenates the information into DMatrix for training ###
xg_train_cv = xgb.DMatrix(x_train_cv, label = y_train_cv) # train set input into xgb
xg_valid = xgb.DMatrix(x_valid, label = y_valid) # valid (test) set input. 

watchlist_cv = [(xg_train_cv, 'train'), (xg_valid, 'valid')]
### 5.3) Runs the model  ###
# start modelling and training on the train and valid df (split from x_train and y_train)
# 
# [500] train-logloss: 0.339 valid-logloss:0.34481 (6 features)
# [500] train-logloss:0.330407  valid-logloss:0.338668 (12 features)
# [499] train-logloss:0.316258  valid-logloss:0.32599 (12 + 7 fuzzywuzzy features)
# [499] train-logloss:0.309309  valid-logloss:0.322939 (eta increased to 0.15)
# [499] train-logloss:0.302894  valid-logloss:0.318291 (24 features, added no space character counts)
# [499] train-logloss:0.294968  valid-logloss:0.310598 (27 features)
#  [499]   train-logloss:0.284764  valid-logloss:0.301273
# (LSA components. Very good results but disappointing score. is it overfitting?)

# [499]   train-logloss:0.279368  valid-logloss:0.29681
# (more LSA features 38 features total)
# [999]   train-logloss:0.207013  valid-logloss:0.225871 (34 features + 4 magic features)
# [999]   train-logloss:0.199112  valid-logloss:0.220251 (38 + 13 abhi features)
# [999]   train-logloss:0.200331 (corrected dataset)
# [999]   train-logloss:0.199891 (55 features + 2 features - hammering dist and shared_2gram = 57 features)
# [999]   train-logloss:0.155211 (57 - 2 hash features + 4 locations + 1 magic feature p2 = 60 features)
# [999]   train-logloss:0.154304 (4 new jelly = 64 features)
# [999]   train-logloss:0.14756 (added q1_q2_intersect_wm and k-core)

# stop iteration if no improvement for 30 rounds 
# where train set improves but test set does not   
bst_cv = xgb.train(params, xg_train_cv, 1000, watchlist_cv, early_stopping_rounds = 30)



################################################################################
################################# 5.C XGBOOST ##################################
################################################################################


# lets train the entire train set 

xg_train = xgb.DMatrix(x_train, label = y_train) # train set input into xgb
 
watchlist = [(xg_train, 'train')]

# train on entire train set 
bst = xgb.train(params, xg_train, 1000, watchlist)


### 5.4) Test the model  ###
# time to input our test dataset into our model 
xg_test = xgb.DMatrix(x_test)
output_result = bst.predict(xg_test)

### 5.5) Write out submission into csv file  ###
# Woof woof
outputsub = pd.DataFrame({'test_id':df_test['test_id'],'is_duplicate':output_result})

outputsub.to_csv('rename_sub.csv',index = False)


################################################################################
################################ 6. FEATURES CHART #############################
################################################################################


variables_important = bst.get_fscore() # dict, check type()
score_df = pd.DataFrame()
score_df['variables'] = variables_important.keys()
score_df['f_score'] = variables_important.values()
score_df.plot(kind= 'barh', x='variables',y='f_score', legend = False)

## Alternatively, run this for better visualization

plt.rcParams['figure.figsize'] = (9.0, 9.0)
xgb.plot_importance(bst); plt.show()
 
