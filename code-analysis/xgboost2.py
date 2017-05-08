
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
# df_train = pd.read_csv('df_train_corrected.csv', encoding = "ISO-8859-1")
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

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
        return '0:0:0:0:0'
    
    # Common non stop words between question pairs. Both variables are equivalent 
    shared_words_q1 = [word for word in q1words.keys() if word in q2words.keys()]
    shared_words_q2 = [word for word in q2words.keys() if word in q1words.keys()]   
    
    
    
    R = (len(shared_words_q1) + len(shared_words_q2))/(len(q1words) + len(q2words))
    R1 = len(q1words) / len(q1) # q1 non stop words ratio
    R2 = len(q2words) / len(q2)
    R3 = R1-R2
    R4 = len(shared_words_q1)
    
    return '{}:{}:{}:{}:{}'.format(R,R1,R2,R3,R4)
    

### 2.2) Set 2: TDIDF (4 features)  ###

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
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





    
################################################################################
############################ 3. FEATURES ENGINEERING ###########################
################################################################################


# Currently 30 features
    # Set 1 (5 features)
    # 1.1 = Proportion of shared words
    # 1.2 = Ratio of q1's non stopwords
    # 1.3 = Ratio of q2's non stopwords
    # 1.4 = Ratio difference (1.3 - 1.4)
    # 1.5 = Length (number) of shared non stop words

    
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
    
    # set 6 (LSA components) - because of the complexity, i will separate set 6 
    # Distance features based on LSA-TFIDF components 
    
    # 6.1 = euclidean 
    # 6.2 = manhattan 
    
    
    
    
    
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

# Set 1
x_train['shared_words'] = temp_df['allR'].apply(lambda x: float(x.split(':')[0]))
x_train['q1_ns_ratio'] = temp_df['allR'].apply(lambda x: float(x.split(':')[1]))
x_train['q2_ns_ratio'] = temp_df['allR'].apply(lambda x: float(x.split(':')[2]))
x_train['ratio_diff'] = temp_df['allR'].apply(lambda x: float(x.split(':')[3]))
x_train['shared_words_length'] = temp_df['allR'].apply(lambda x: float(x.split(':')[4]))


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

del temp_df

################################################################################
################################################################################

################################## TESTING SET ################################

# create temp df for set 1 function
temp_df_test['allR'] = df_test.apply(shared_words, axis = 1, raw = True)
temp_df_test['tfidf_all'] = df_test.apply(tfidf_word_match_share, axis = 1, raw = True)

# Set 1 
x_test['shared_words'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[0]))
x_test['q1_ns_ratio'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[1]))
x_test['q2_ns_ratio'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[2]))
x_test['ratio_diff'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[3]))
x_test['shared_words_length'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[4]))

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

import pandas as pd 
import numpy as np



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
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

train_qs = train_qs.tolist()
vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english')
dtm = vectorizer.fit_transform(train_qs) 



lsa = TruncatedSVD(2, algorithm = 'randomized')
dtm_lsa = lsa.fit_transform(dtm)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

# lets view 
pd.DataFrame(dtm_lsa, index = train_qs, columns = ["component_1","component_2"])

# convert to DF
temp_component_train = pd.DataFrame(dtm_lsa, columns = ["component_1","component_2"])
  
# now we need to split them 
temp_component_train_q1 = temp_component_train[:404290]
temp_component_train_q2 = temp_component_train[404290:808580]
temp_component_train_q2.index = range(404290)

# recombine them 
temp_train_vector = pd.DataFrame()
temp_train_vector['q1_c1'] = temp_component_train_q1['component_1']
temp_train_vector['q1_c2'] = temp_component_train_q1['component_2']
temp_train_vector['q2_c1'] = temp_component_train_q2['component_1']
temp_train_vector['q2_c2'] = temp_component_train_q2['component_2']

# remove 
del temp_component_train_q1, temp_component_train_q2, temp_component_train

# lets do some distancing features  
distances = pd.DataFrame()
distances['euclidean'] = temp_train_vector.apply(euclidean_distance, axis = 1)
distances['manhattan'] = temp_train_vector.apply(manhattan_distance, axis = 1)

distances.to_csv('lsa_distance.csv',index = False)

# do for test set 

test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

test_qs = test_qs.tolist()
vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english')
dtm2 = vectorizer.fit_transform(test_qs)

lsa = TruncatedSVD(2, algorithm = 'randomized')
dtm_lsa2 = lsa.fit_transform(dtm2)
dtm_lsa2 = Normalizer(copy=False).fit_transform(dtm_lsa2)
temp_component_test = pd.DataFrame(dtm_lsa2, columns = ["component_1","component_2"])

# now we need to split them 
temp_component_test_q1 = temp_component_test[:2345796]
temp_component_test_q2 = temp_component_test[2345796:4691592]
temp_component_test_q2.index = range(2345796)

temp_test_vector = pd.DataFrame()
temp_test_vector['q1_c1'] = temp_component_test_q1['component_1']
temp_test_vector['q1_c2'] = temp_component_test_q1['component_2']
temp_test_vector['q2_c1'] = temp_component_test_q2['component_1']
temp_test_vector['q2_c2'] = temp_component_test_q2['component_2']

del temp_component_test, temp_component_test_q1, temp_component_test_q2

# lets do some distancing features  
distances_test = pd.DataFrame()
distances_test['euclidean'] = temp_test_vector.apply(euclidean_distance, axis = 1)
distances_test['manhattan'] = temp_test_vector.apply(manhattan_distance, axis = 1)


# add back to x_train and x_test as features 



# new features 
x_train['euclidean'] = distances['euclidean']
x_train['manhattan'] = distances['manhattan']

x_test['euclidean'] = distances_test['euclidean']
x_test['manhattan'] = distances_test['manhattan']
 

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


x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = random)


################################################################################
################################# 5. XGBOOST MODEL #############################
################################################################################

### 5.1) Setting the model parameters  ###
params = {} # dict 
params['eta'] = 0.15
params['max_depth'] = 5 
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['seed'] = random


### 5.2) Concatenates the information into DMatrix for training ###
xg_train = xgb.DMatrix(x_train, label = y_train) # train set input into xgb
xg_valid = xgb.DMatrix(x_valid, label = y_valid) # valid (test) set input. 

watchlist = [(xg_train, 'train'), (xg_valid, 'valid')]


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

# stop iteration if no improvement for 30 rounds 
# where train set improves but test set does not    
bst = xgb.train(params, xg_train, 500, watchlist, early_stopping_rounds = 30)


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

plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(bst); plt.show()
 