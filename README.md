---
title: "Entry for Kaggle - Quora Competition"
author: "Team Members: Germayne Ng, Alson Yap"
---

![quora](https://cloud.githubusercontent.com/assets/22788747/24694479/a7783804-1a14-11e7-8589-40641ffdeb93.png)


# Competition 
<br>
Link to competition details and question: https://www.kaggle.com/c/quora-question-pairs

---

# Update Logs
<br>

* **Version 1.6 - 13th May 2017:**
  
  + Revamp LSA features, based on train and test data as 'document' as opposed to referencing them seperately. 
  + 6 new features: LSA Q1 component 1, 2 , Q2 component 1, 2 and 2 distancing based on the components 
  + total 55. (note that 2 features will overwrite the old distancing features, so 4 new additions)
  + Major change in xgboost.py script: ** I have split section 5 into a,b,c. For Cross validation, do 5a> 5b. For modelling to submit, do 5a > 5c, skip 5b. Mainly to use full training set to modelling. **
  + score - 0.27XX
  
* **Version 1.6 - 12th May 2017:**

  + Added 4 magic features - 0.30XX 
  + Added AB 12 features - 0.28XX 
  + Total 51 features 
  
* **Version 1.5 - 20th April 2017:**

  + Finally best score so far. Managed to make good use of the LSA components features. 
  + Distance features can be further distingused. Alson did the distance for **single** vector. If you define LSA components and apply      distance, it can be a seperate feature. 
  + Based on Alson's functions for distance, I created a euclidean and manhattan function for **single** vector. Essentially, there are     2 features based on euclidean and manhattan distances (total 4 )  
  
  
  LSA components: 
  Basically each question is now a vector in LSA-TFIDF components. I.E. Note the values are arbitary, just for example sake:  
  
  question | component 1  | component 2 | question | component 1 | component 2
  --- | --- | --- | --- | --- | ---
  question 1 of pair 1 | 0.23 | 0.56 | question 2 of pair 1 | 0.4 | 0.7 
  
  So, question 1 instead of words is now [0.23 , 0.56] and question 2 [0.4, 0.7].
  Now as vectors, we then calculate its distances.
  
  + tune 1000 nrounds and lower learning rate to 0.1 = better results.
  
* **Version 1.4 - 4th April 2017:**

  + Expanded on the TFIDF function (3), Added character count **without** spaces (3) and character count per word (3) 
  + Total 30 features. Accuracy: 0.32624 (Rank 162: Top 14%)
  + Updated features dataset in dropbox

* **Version 1.3 - 1st April 2017:**

  + Added Jaccard distance and Cosine distance features. Total 21 features. (Rank 133: Top 15%)

* **Version 1.2 - 1st April 2017:**

  + Added 7 FuzzyWuzzy features. Total 19 features. (Rank 151: Top 15%) 

* **Version 1.1 - 31st March 2017:**

  + Added additional features. Total 12 features. (Rank: 215 Top 25%)

* **Version 1.0 - 30th March 2017:**

  + Implemented Xgboost with 6 features.  

---

# References 

![f81e92c0-472b-4770-b808-1abdd9376edf-original](https://cloud.githubusercontent.com/assets/22788747/24948646/85467898-1f9d-11e7-8d68-cdc03a9e9a9e.png)


* Initial reference: https://www.kaggle.com/alijs1/quora-question-pairs/xgb-starter-12357/code

* FuzzyWuzzy reference: 

    + https://www.kaggle.com/aneeshc/quora-question-pairs/fuzzy-feature-based-classification/notebook

    + https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur

* Jaccard and Cosine distance: https://www.kaggle.com/heraldxchaos/quora-question-pairs/adventures-in-scikitlearn-and-nltk/run/1040772
    
* Future reference for new ideas:https://www.kaggle.com/c/quora-question-pairs/discussion/30340#171996

* SVD, LSA components 
    + http://mccormickml.com/2016/03/25/lsa-for-text-classification-tutorial/
    + http://www.datascienceassn.org/sites/default/files/users/user1/lsa_presentation_final.pdf

* Xgboost references
    + https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    
    + http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html
    
* The art of hyper tuning 
    + https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    
    + https://www.slideshare.net/ShangxuanZhang/kaggle-winning-solution-xgboost-algorithm-let-us-learn-from-its-author
    
    + https://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1
    
* Xgboost notes by germ: 
    + https://github.com/germayneng/xgboost-notes/blob/master/README.md
    
* Ensemble methods: 
    + https://mlwave.com/kaggle-ensembling-guide/
---

# To do/ comments

1. **Running the spell checker script on train set and testing set (~20k per hour)**

So, train set will take about 404/20 = ~20 hours.  
Then, test set will take 20hours * 6 = 120 hours == 5 days +

Spell checker script taken from: http://norvig.com/spell-correct.html

Download big.txt file because the script needs to reference to that corpus of words.
I only added the sentence_correction function which is to be run on both data sets.


**Conclusion: Fuzzy wuzzy features are really significant. But realise that after implementing the Jaccard and Cosine dist features, the rankings changed. Refer to the features importance in the figures folder**

---
