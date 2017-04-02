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


#### 1.2) Read the data ###

# For reading in the spell corrected df_train file
# df_train = pd.read_csv('df_train_corrected.csv', encoding = "ISO-8859-1")
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

################################################################################
############################## 2. DEFINE FUNCTIONS #############################
################################################################################

### 2.1) Function 1: Shared words (5 features)  ###

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
    
    # Common non stop words between question pairs. Both variableS are equivalent 
    shared_words_q1 = [word for word in q1words.keys() if word in q2words.keys()]
    shared_words_q2 = [word for word in q2words.keys() if word in q1words.keys()]   
    
    
    
    R = (len(shared_words_q1) + len(shared_words_q2))/(len(q1words) + len(q2words))
    R1 = len(q1words) / len(q1) # q1 non stop words ratio
    R2 = len(q2words) / len(q2)
    R3 = R1-R2
    R4 = len(shared_words_q1)
    
    return '{}:{}:{}:{}:{}'.format(R,R1,R2,R3,R4)
    

### 2.2) Function 2: TDIDF (1 feature)  ###

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

### 2.3) Function 3: Jaccard Distance (1 feature)  ###

def jaccard_dist(row):
    return jaccard_distance(set(str(row['question1'])), set(str(row['question2'])))
    
### 2.4) Function 4: Cosine Distance (1 feature)  ###

def cosine_dist(row):
    a = set(str(row['question1']))
    b = set(str(row['question2']))
    d = len(a)*len(b)
    if (d == 0):
        return 0
    else: 
        return len(a.intersection(b))/d
    
    
################################################################################
############################ 3. FEATURES ENGINEERING ###########################
################################################################################


# Currently 19 features
    # Set 1 (5 features)
    # 1.1 = Proportion of shared words
    # 1.2 = Ratio of q1's non stopwords
    # 1.3 = Ratio of q2's non stopwords
    # 1.4 = Ratio difference (1.3 - 1.4)
    # 1.5 = Length (number) of shared words
    
    #Set 2 (1 feature)
    # 2.1 = TFIDF 
    
    # Set 3 (6 features)
    # 3.1 = Word count in q1
    # 3.2 = Word count in q2
    # 3.3 = Word count difference 
    # 3.4 = Character count in q1 (including spaces)
    # 3.5 = Character count in q2 (including spaces)
    # 3.6 = Character count difference (including spaces)
    
    # Set 4 (7 features - FuzzyWuzzy)
    # 4.1 = QRatio
    # 4.2 = WRatio
    # 4.3 = Partial ratio
    # 4.4 = Partial token set ratio
    # 4.5 = Partial token sort ratio
    # 4.6 = Token set ratio
    # 4.7 = Token sort ratio
    
    # Set 5 (2 features) (Potential to add more under this set!)
    # 5.1 = Jaccard distance
    # 5.2 = Cosine distance

### 3.1) Creation of dataframes for training and testing  ###

x_train = pd.DataFrame() # hold the training set 
x_test = pd.DataFrame() # hold the testing set 
temp_df = pd.DataFrame() # to be removed later
temp_df_test = pd.DataFrame()

### 3.2) Generating and loading the features  ###
    
################################## TRAINING SET ################################

# create temp df for set 1 function  
temp_df['allR'] = df_train.apply(shared_words, axis = 1, raw = True)

# Set 1
x_train['shared_words'] = temp_df['allR'].apply(lambda x: float(x.split(':')[0]))
x_train['q1_ns_ratio'] = temp_df['allR'].apply(lambda x: float(x.split(':')[1]))
x_train['q2_ns_ratio'] = temp_df['allR'].apply(lambda x: float(x.split(':')[2]))
x_train['ratio_diff'] = temp_df['allR'].apply(lambda x: float(x.split(':')[3]))
x_train['shared_words_length'] = temp_df['allR'].apply(lambda x: float(x.split(':')[4]))

# Set 2
x_train['tfidf'] = df_train.apply(tfidf_word_match_share, axis = 1, raw = True)

# Set 3
x_train['q1_word_count'] = df_train['question1'].apply(lambda x: len(str(x).lower().split()))
x_train['q2_word_count'] = df_train['question2'].apply(lambda x: len(str(x).lower().split()))
x_train['diff_word_count'] = x_train['q1_word_count'] - x_train['q2_word_count']

x_train['q1_char_count_withspace'] = df_train['question1'].apply(lambda x: len(str(x)))
x_train['q2_char_count_withspace'] = df_train['question2'].apply(lambda x: len(str(x)))
x_train['diff_char_count_withspace'] = x_train['q1_char_count_withspace'] - x_train['q2_char_count_withspace']

# Set 4
x_train['fuzz_qratio'] = df_train.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_WRatio'] = df_train.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_partial_ratio'] = df_train.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_partial_token_set_ratio'] = df_train.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_partial_token_sort_ratio'] = df_train.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_token_set_ratio'] = df_train.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_train['fuzz_token_sort_ratio'] = df_train.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

# Set 5
x_train['jaccard_dist'] = df_train.apply(jaccard_dist, axis = 1)
x_train['cosine_dist'] = df_train.apply(cosine_dist, axis = 1)

del temp_df

################################################################################
################################################################################

################################## TESTING SET ################################

# create temp df for set 1 function
temp_df_test['allR'] = df_test.apply(shared_words, axis = 1, raw = True)

# Set 1 
x_test['shared_words'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[0]))
x_test['q1_ns_ratio'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[1]))
x_test['q2_ns_ratio'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[2]))
x_test['ratio_diff'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[3]))
x_test['shared_words_length'] = temp_df_test['allR'].apply(lambda x: float(x.split(':')[4]))

# Set 2
x_test['tfidf'] = df_test.apply(tfidf_word_match_share, axis = 1, raw = True)

# Set 3
x_test['q1_word_count'] = df_test['question1'].apply(lambda x: len(str(x).lower().split()))
x_test['q2_word_count'] = df_test['question2'].apply(lambda x: len(str(x).lower().split()))
x_test['diff_word_count'] = x_test['q1_word_count'] - x_test['q2_word_count']

x_test['q1_char_count_withspace'] = df_test['question1'].apply(lambda x: len(str(x)))
x_test['q2_char_count_withspace'] = df_test['question2'].apply(lambda x: len(str(x)))
x_test['diff_char_count_withspace'] = x_test['q1_char_count_withspace'] - x_test['q2_char_count_withspace']

# Set 4
x_test['fuzz_qratio'] = df_test.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_WRatio'] = df_test.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_partial_ratio'] = df_test.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_partial_token_set_ratio'] = df_test.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_partial_token_sort_ratio'] = df_test.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_token_set_ratio'] = df_test.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
x_test['fuzz_token_sort_ratio'] = df_test.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

# Set 5
x_test['jaccard_dist'] = df_test.apply(jaccard_dist, axis = 1)
x_test['cosine_dist'] = df_test.apply(cosine_dist, axis = 1)


# remove temp 
del temp_df_test 

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

random = 12357
np.random.seed(random)
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
params['seed'] = 123


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
bst = xgb.train(params, xg_train, 500, watchlist)


### 5.4) Test the model  ###
# time to input our test dataset into our model 
xg_test = xgb.DMatrix(x_test)
output_result = bst.predict(xg_test)

### 5.5) Write out submission into csv file  ###
# Woof woof
nextsub = pd.DataFrame({'test_id':df_test['test_id'],'is_duplicate':output_result})
nextsub.to_csv('nextsub_descriptionhere.csv',index = False)


################################################################################
################################ 6. FEATURES CHART #############################
################################################################################


variables_important = bst.get_fscore() # dict
score_df = pd.DataFrame()
score_df['variables'] = variables_important.keys()
score_df['f_score'] = variables_important.values()
score_df.plot(kind= 'barh', x='variables',y='f_score', legend = False)

## Alternatively, run this for better visualization

plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(bst); plt.show()