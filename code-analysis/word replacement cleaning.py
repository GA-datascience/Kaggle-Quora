import pandas as pd
import numpy as np
import nltk
import re
from string import punctuation



df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')



def text_to_wordlist(text):
    
    # Replace weird chars in text
    
    text = re.sub("’", "'", text) # special single quote
    text = re.sub("`", "'", text) # special single quote
    text = re.sub("“", '"', text) # special double quote
    text = re.sub("？", "?", text) 
    text = re.sub("…", " ", text) 
    text = re.sub("é", "e", text) 
    
    # Clean shorthands
    text = re.sub("\'s", " ", text) # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("\'re", " are ", text)
    text = re.sub("can't", "can not ", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am ", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub(r"(\W|^)([0-9]+)[kK](\W|$)", r"\1\g<2>000\3", text) # better regex provided by @armamut
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)
    
    
    # Remove comma between numbers, i.e. 15,000 -> 15000
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    
    # Add padding to punctuations and special chars
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    
    # indian dollar
    
    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)
    
    # Clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text) 
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)  
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE) 
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)
    
    # Typos identified with my eyes
    
    text = re.sub(r" quikly ", " quickly ", text)
    text = re.sub(r" unseccessful ", " unsuccessful ", text)
    text = re.sub(r" demoniti[\S]+ ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" demoneti[\S]+ ", " demonetization ", text, flags=re.IGNORECASE)  
    text = re.sub(r" addmision ", " admission ", text)
    text = re.sub(r" insititute ", " institute ", text)
    text = re.sub(r" connectionn ", " connection ", text)
    text = re.sub(r" permantley ", " permanently ", text)
    text = re.sub(r" sylabus ", " syllabus ", text)
    text = re.sub(r" sequrity ", " security ", text)
    text = re.sub(r" undergraduation ", " undergraduate ", text) # not typo, but GloVe can't find it
    text = re.sub(r"(?=[a-zA-Z])ig ", "ing ", text)
    text = re.sub(r" latop", " laptop", text)
    text = re.sub(r" programmning ", " programming ", text)  
    text = re.sub(r" begineer ", " beginner ", text)  
    text = re.sub(r" qoura ", " Quora ", text)
    text = re.sub(r" wtiter ", " writer ", text)  
    text = re.sub(r" litrate ", " literate ", text) 
    
    # The single 's' in this stage is 99% of not clean text, just kill it
    text = re.sub(' s ', " ", text)
    
    # Reduce extra spaces into single spaces
    text = re.sub('[\s]+', " ", text)
    text = text.strip()

    
    # Return a list of words
    return(text)


# Function for concatenating all the required processes into one
def process_questions(question_list, questions, question_list_name, dataframe):
    '''transform questions and display progress'''
    for question in questions:
        question_list.append(text_to_wordlist(question))
        if len(question_list) % 100000 == 0:
            progress = len(question_list)/len(dataframe) * 100
            print("{} is {}% complete.".format(question_list_name, round(progress, 1)))
            
            
            
# Execution            
train_question1 = []
process_questions(train_question1, df_train.question1, 'train_question1', train)

train_question2 = []
process_questions(train_question2, df_train.question2, 'train_question2', train)

test_question1 = []
process_questions(test_question1, df_test.question1, 'test_question1', test)

test_question2 = []
process_questions(test_question2, df_test.question2, 'test_question2', test)