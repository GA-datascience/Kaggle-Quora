# xgboost 
# 

# import the standard libraries

import numpy as np # lA 
import pandas as pd # data process 
#import matplotlib.pyplot as plt # for graph 

from nltk.corpus import stopwords 
from collections import Counter 
from fuzzywuzzy import fuzz

from sklearn.cross_validation import train_test_split 
import xgboost as xgb # xgboost model 



# import data files 

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')



# functions 
# 2 functions for nonstop words comparison as well as TFIDF using weights 



#   function 1: Shared words 
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
    
    # common non stop words between question pairs. Both variable are equivalent 
    shared_words_q1 = [word for word in q1words.keys() if word in q2words.keys()]
    shared_words_q2 = [word for word in q2words.keys() if word in q1words.keys()]   
    
    
    
    R = (len(shared_words_q1) + len(shared_words_q2))/(len(q1words) + len(q2words))
    R1 = len(q1words) / len(q1) #q1 non stop words ratio
    R2 = len(q2words) / len(q2)
    R3 = R1-R2
    R4 = len(shared_words_q1)
    
    return '{}:{}:{}:{}:{}'.format(R,R1,R2,R3,R4)
    


#   function 2: TFIDF 

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
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    

    shared_weights = [weights.get(word,0) for word in q1words if word in q2words] + [weights.get(word,0) for word in q2words if word in q1words]   
    total_weights = [weights.get(word,0) for word in q1words] + [weights.get(word,0) for word in q2words]
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

#   function 3: Jaccard Distance 

def jaccard_dist(row):
    return jaccard_distance(set(str(row['question1'])), set(str(row['question2'])))
    
#   function 4: Cosine Distance

def cosine_dist(row):
    a = set(str(row[question1']))
    b = set(str(row[question2']))
    d = len(a)*len(b)
    if (d == 0):
        return 0
    else: 
        return len(a.intersection(b))/d
    
    

# features Engineering
# currently: 12 features
    # Shared words proportion 
    # TFIDF 
    
    # q1's non stop words ratio 
    # q2's non stop words ratio
    # question pair ratio difference 
    
    # number of shared words in question pairs 
    
    # word count in q1
    # word count in q2
    # word count difference 
    
    # character count in q1 (including spaces)
    # character count in q2 (including spaces)
    # character count difference (including spaces)



x_train = pd.DataFrame() # for train set 
x_test = pd.DataFrame() # for test set 
temp_df = pd.DataFrame() # later remove 
temp_df_test = pd.DataFrame()

# filling in the features. train set  
temp_df['allR'] = df_train.apply(shared_words, axis = 1, raw = True)

x_train['shared_words'] = temp_df['allR'].apply(lambda x: float(x.split(':')[0]))
x_train['q1_ns_ratio'] = temp_df['allR'].apply(lambda x: float(x.split(':')[1]))
x_train['q2_ns_ratio'] = temp_df['allR'].apply(lambda x: float(x.split(':')[2]))
x_train['ratio_diff'] = temp_df['allR'].apply(lambda x: float(x.split(':')[3]))

x_train['shared_words_length'] = temp_df['allR'].apply(lambda x: float(x.split(':')[4]))

x_train['tfidf'] = df_train.apply(tfidf_word_match_share, axis = 1, raw = True)

x_train['q1_word_count'] = df_train['question1'].apply(lambda x: len(str(x).lower().split()))
x_train['q2_word_count'] = df_train['question2'].apply(lambda x: len(str(x).lower().split()))
x_train['diff_word_count'] = x_train['q1_word_count'] - x_train['q2_word_count']

x_train['q1_char_count_withspace'] = df_train['question1'].apply(lambda x: len(str(x)))
x_train['q2_char_count_withspace'] = df_train['question2'].apply(lambda x: len(str(x)))
x_train['diff_char_count_withspace'] = x_train['q1_char_count_withspace'] - x_train['q2_char_count_withspace']

x_train['fuzz_qratio'] = df_train.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_WRatio'] = df_train.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_partial_ratio'] = df_train.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_partial_token_set_ratio'] = df_train.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_partial_token_sort_ratio'] = df_train.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_token_set_ratio'] = df_train.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_token_sort_ratio'] = df_train.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

del temp_df

# filling in the features. test set 

temp_df_test['allR'] = df_test.apply(shared_words, axis = 1, raw = True)
x_test['shared_words'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[0]))


x_test['q1_ns_ratio'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[1]))
x_test['q2_ns_ratio'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[2]))
x_test['ratio_diff'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[3]))

x_test['shared_words_length'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[4]))

x_test['tfidf'] = df_test.apply(tfidf_word_match_share, axis = 1, raw = True)

x_test['q1_word_count'] = df_test['question1'].apply(lambda x: len(str(x).lower().split()))
x_test['q2_word_count'] = df_test['question2'].apply(lambda x: len(str(x).lower().split()))
x_test['diff_word_count'] = x_test['q1_word_count'] - x_test['q2_word_count']

x_test['q1_char_count_withspace'] = df_test['question1'].apply(lambda x: len(str(x)))
x_test['q2_char_count_withspace'] = df_test['question2'].apply(lambda x: len(str(x)))
x_test['diff_char_count_withspace'] = x_test['q1_char_count_withspace'] - x_test['q2_char_count_withspace']

x_test['fuzz_qratio'] = df_test.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)

x_test['fuzz_WRatio'] = df_test.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_partial_ratio'] = df_test.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_partial_token_set_ratio'] = df_test.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_partial_token_sort_ratio'] = df_test.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_token_set_ratio'] = df_test.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_token_sort_ratio'] = df_test.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


# remove temp 
del temp_df_test 


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

random = 12357
np.random.seed(random)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = random)




# XGBOOST 

params = {} # dict 
params['eta'] = 0.11
params['max_depth'] = 5 
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['seed'] = random

xg_train = xgb.DMatrix(x_train, label = y_train) # train set input into xgb
xg_valid = xgb.DMatrix(x_valid, label = y_valid) # valid (test) set input. 

watchlist = [(xg_train, 'train'), (xg_valid, 'valid')]


# start modelling and training on the train and valid df (split from x_train and y_train)
# 
# [500] train-logloss: 0.339 valid-logloss:0.34481 (6 features)
# [500] train-logloss:0.330407  valid-logloss:0.338668 (12 features)
bst = xgb.train(params, xg_train, 500, watchlist)


# time to input our test dataset into our model 


xg_test = xgb.DMatrix(x_test)
output_result = bst.predict(xg_test)

# excited to see our fourth submission, will it be improvement?
nextsub = pd.DataFrame({'test_id':df_test['test_id'],'is_duplicate':output_result})
nextsub.to_csv('fifthsub.csv',index = False)



# Charts 

# plotting the important variables 

variables_important = bst.get_fscore() # dict
score_df = pd.DataFrame()
score_df['variables'] = variables_important.keys()
score_df['f_score'] = variables_important.values()
score_df.plot(kind= 'barh', x='variables',y='f_score', legend = False)




# References 
# 
# https://www.kaggle.com/alijs1/quora-question-pairs/xgb-starter-12357/code




