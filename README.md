---
title: "Entry for Kaggle - Quora Competition"
author: "Team Members: Germayne Ng, Alson Yap"
---

# Update Logs
<br>

* **Version 1.3 - 1st April 2017:**

Added Jaccard distance and Cosine distance features. Total 21 features. (Rank 133: Top 15%)

* **Version 1.2 - 1st April 2017:**

Added 7 FuzzyWuzzy features. Total 19 features. (Rank 151: Top 15%) 

* **Version 1.1 - 31st March 2017:**

Added additional features. Total 12 features. (Rank: 215 Top 25%)

* **Version 1.0 - 30th March 2017:**

Implemented Xgboost with 6 features.  

# References 

* Initial reference: https://www.kaggle.com/alijs1/quora-question-pairs/xgb-starter-12357/code

* FuzzyWuzzy reference: 

    + https://www.kaggle.com/aneeshc/quora-question-pairs/fuzzy-feature-based-classification/notebook

    + https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur
    
* Future reference for new ideas:https://www.kaggle.com/c/quora-question-pairs/discussion/30340#171996


# For Germ's reference on coming Monday (delete after read)
To describe what I have done over the weekend

1. **Running the spell checker script on train set and testing set (~20k per hour)**

So, train set will take about 404/20 = ~20 hours.  
Then, test set will take 20hours * 6 = 120 hours == 5 days +

Spell checker script taken from: http://norvig.com/spell-correct.html

Download big.txt file because the script needs to reference to that corpus of words.
I only added the sentence_correction function which is to be run on both data sets.

2. **Added fuzzy wuzzy features**

Refer to the links above.

Also suggest to check out the fuzzywuzzy's Github page for information on those functions.
Interesting concept and easy to understand examples.

Implementation takes very long, so do not suggest running again.

3. **Added Jaccard and Cosine distance features**

Refer to: https://www.kaggle.com/heraldxchaos/quora-question-pairs/adventures-in-scikitlearn-and-nltk/run/1040772

These 2 features can also be found in the Indian dude's script.

4. **Created Dropbox account**

Download the x_test.csv and x_train.csv with all the 21 features loaded inside.
Can just load and add in new features from there onwards.

**Conclusion: Fuzzy wuzzy features are really significant. But realise that after implementing the Jaccard and Cosine dist features, the rankings changed. Refer to the features importance in the figures folder**
