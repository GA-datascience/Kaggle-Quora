# LSA components with TFIDF vectors 

# http://www.datascienceassn.org/sites/default/files/users/user1/lsa_presentation_final.pdf


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
temp_component_train = pd.DataFrame(dtm_lsa, columns = ["component_1","component_2"])

# now we need to split them 
temp_component_train_q1 = temp_component_train[:404290]
temp_component_train_q2 = temp_component_train[404290:808580]
temp_component_train_q2.index = range(404290)



x_train['lsa_tfidf_c1_q1'] = temp_component_train_q1['component_1']
x_train['lsa_tfidf_c2_q1'] = temp_component_train_q1['component_2']
x_train['lsa_tfidf_c1_q2'] = temp_component_train_q2['component_1']
x_train['lsa_tfidf_c2_q2'] = temp_component_train_q2['component_2']



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



x_test['lsa_tfidf_c1_q1'] = temp_component_test_q1['component_1']
x_test['lsa_tfidf_c2_q1'] = temp_component_test_q1['component_2']
x_test['lsa_tfidf_c1_q2'] = temp_component_test_q2['component_1']
x_test['lsa_tfidf_c2_q2'] = temp_component_test_q2['component_2']

