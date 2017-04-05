---
title: "Entry for Kaggle - Quora Competition"
author: "Team Members: Germayne Ng, Alson Yap"
---
![quora](https://cloud.githubusercontent.com/assets/22788747/24694479/a7783804-1a14-11e7-8589-40641ffdeb93.png)

# Competition 
<br>
Link to competition details and question: https://www.kaggle.com/c/quora-question-pairs

# Update Logs
<br>

* **Version 1.4 - 4th April 2017:**

 + Expanded on the TFIDF function (3), Added character count **without** spaces (3) and character count per word (3) 
 + Total 30 features. Accuracy: 0.32624 (Rank 162: Top 14%)
 + Updated features dataset in dropbox

* **Version 1.3 - 1st April 2017:**

Added Jaccard distance and Cosine distance features. Total 21 features. (Rank 133: Top 15%)

* **Version 1.2 - 1st April 2017:**

 + Added 7 FuzzyWuzzy features. Total 19 features. (Rank 151: Top 15%) 

* **Version 1.1 - 31st March 2017:**

 + Added additional features. Total 12 features. (Rank: 215 Top 25%)

* **Version 1.0 - 30th March 2017:**

 + Implemented Xgboost with 6 features.  

---

# References 

* Initial reference: https://www.kaggle.com/alijs1/quora-question-pairs/xgb-starter-12357/code

* FuzzyWuzzy reference: 

    + https://www.kaggle.com/aneeshc/quora-question-pairs/fuzzy-feature-based-classification/notebook

    + https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur

* Jaccard and Cosine distance: https://www.kaggle.com/heraldxchaos/quora-question-pairs/adventures-in-scikitlearn-and-nltk/run/1040772
    
* Future reference for new ideas:https://www.kaggle.com/c/quora-question-pairs/discussion/30340#171996

* Xgboost references
    + https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    
    + http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html
    
* The art of hyper tuning 
    + https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    
    + https://www.slideshare.net/ShangxuanZhang/kaggle-winning-solution-xgboost-algorithm-let-us-learn-from-its-author
    
    + https://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1
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
