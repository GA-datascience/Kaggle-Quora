import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict

train_orig =  pd.read_csv('train_corrected.csv', header=0, encoding = "ISO-8859-1")
test_orig =  pd.read_csv('test_corrected.csv', header=0, encoding = "ISO-8859-1")

# "id","qid1","qid2","question1","question2","is_duplicate"
df_id1 = train_orig[["qid1", "question1"]].drop_duplicates(keep="first").copy().reset_index(drop=True)

df_id2 = train_orig[["qid2", "question2"]].drop_duplicates(keep="first").copy().reset_index(drop=True)

df_id1.columns = ["qid", "question"]
df_id2.columns = ["qid", "question"]

print(df_id1.shape, df_id2.shape)

df_id = pd.concat([df_id1, df_id2]).drop_duplicates(keep="first").reset_index(drop=True)
print(df_id1.shape, df_id2.shape, df_id.shape)


import csv
dict_questions = df_id.set_index('question').to_dict()
dict_questions = dict_questions["qid"]

new_id = 538000 # df_id["qid"].max() ==> 537933

def get_id(question):
    global dict_questions 
    global new_id 
    
    if question in dict_questions:
        return dict_questions[question]
    else:
        new_id += 1
        dict_questions[question] = new_id
        return new_id

rows = []
max_lines = 10
if True:
    with open('test_corrected.csv', 'r', encoding="ISO-8859-1") as infile:
        reader = csv.reader(infile, delimiter=",")
        header = next(reader)
        header.append('qid1')
        header.append('qid2')
        
        if True:
            print(header)
            pos, max_lines = 0, 10*1000*1000
            for row in reader:
                # "test_id","question1","question2"
                question1 = row[1]
                question2 = row[2]

                qid1 = get_id(question1)
                qid2 = get_id(question2)
                row.append(qid1)
                row.append(qid2)

                pos += 1
                if pos >= max_lines:
                    break
                rows.append(row)
                
test_with_ids = pd.DataFrame(data=rows, columns=header)
test_with_ids.to_csv('test_with_ids.csv', index=False)

####################################################################
#### Part 2 of the kernel ### Can clear kernel first before doing this ###

import networkx as nx

df_train = pd.read_csv('train_corrected.csv', usecols=["qid1", "qid2"])
df_test = pd.read_csv('test_with_ids.csv', usecols=["qid1", "qid2"])
df_all = pd.concat([df_train, df_test])
print("df_all.shape:", df_all.shape) # df_all.shape: (2750086, 2)

df = df_all

g = nx.Graph()
g.add_nodes_from(df.qid1)

edges = list(df[['qid1', 'qid2']].to_records(index=False))

g.add_edges_from(edges)

g.remove_edges_from(g.selfloop_edges())

print(len(set(df.qid1)), g.number_of_nodes()) # 4789604

print(len(df), g.number_of_edges()) # 2743365 (after self-edges)

df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])

print("df_output.shape:", df_output.shape)

NB_CORES = 20

for k in range(2, NB_CORES + 1):

    fieldname = "kcore{}".format(k)

    print("fieldname = ", fieldname)

    ck = nx.k_core(g, k=k).nodes()

    print("len(ck) = ", len(ck))

    df_output[fieldname] = 0

    df_output.ix[df_output.qid.isin(ck), fieldname] = k

df_output.to_csv("question_kcores.csv", index=None)

####### Finds the max core and summarize a dataframe with qid and its corresponding max core value

df_cores = pd.read_csv("question_kcores.csv", index_col="qid")

df_cores.index.names = ["qid"]

df_cores['max_kcore'] = df_cores.apply(lambda row: max(row), axis=1)

df_cores[['max_kcore']].to_csv("question_max_kcores.csv") # with index



###### Additional part to fill up the max kcore in x_train and x_test

cores_dict = pd.read_csv("question_max_kcores.csv", index_col="qid").to_dict()["max_kcore"]
def gen_qid1_max_kcore(row):
    return cores_dict[row["qid1"]]
def gen_qid2_max_kcore(row):
    return cores_dict[row["qid2"]]


df_train = pd.read_csv('train_corrected.csv', usecols=["qid1", "qid2"])
df_test = pd.read_csv('test_with_ids.csv', usecols=["qid1", "qid2"])

df_train["qid1_max_kcore"] = df_train.apply(gen_qid1_max_kcore, axis=1)
df_train["qid2_max_kcore"] = df_train.apply(gen_qid2_max_kcore, axis=1)

df_test["qid1_max_kcore"] = df_test.apply(gen_qid1_max_kcore, axis=1)
df_test["qid2_max_kcore"] = df_test.apply(gen_qid2_max_kcore, axis=1)


###### Transfer the 2 features over to x_train and x_test


x_train = pd.read_csv('x_train.csv')
x_test = pd.read_csv('x_test.csv')

x_train["qid1_max_kcore"] = df_train["qid1_max_kcore"]
x_train["qid2_max_kcore"] = df_train["qid2_max_kcore"]

x_test["qid1_max_kcore"] = df_test["qid1_max_kcore"]
x_test["qid2_max_kcore"] = df_test["qid2_max_kcore"]

x_train.to_csv("x_train_kcore.csv", index=False)
x_test.to_csv("x_test_kcore.csv", index=False)












