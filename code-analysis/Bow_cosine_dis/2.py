import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ParamConfig:
    def __init__(self,
                 data_path,
                 processed_data_path,
                 stemmer_type):
        self.data_path = data_path
        self.processed_data_path = processed_data_path
        self.stemmer_type = stemmer_type

config = ParamConfig(data_path = "./data",
                     processed_data_path = "./processed_data",
                     stemmer_type = "snowball")



#################
## Calculation ##
#################
def try_divide(x, y, val=0.0):
    if y != 0.0:
        val = float(x) / y
    return val

def cat_text(x):
    res = '%s %s' % (x['question1'], x['question2'])
    return res


################
## Stop Words ##
################
stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)


##############
## Stemming ##
##############
if config.stemmer_type == "porter":
    english_stemmer = nltk.stem.PorterStemmer()
elif config.stemmer_type == "snowball":
    english_stemmer = nltk.stem.SnowballStemmer('english')
def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed


#####################
## Distance metric ##
#####################
def JaccardCoef(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef = try_divide(intersect, union)
    return coef

def DiceDist(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = try_divide(2*intersect, union)
    return d

def compute_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = JaccardCoef(A, B)
    elif dist == "dice_dist":
        d = DiceDist(A, B)
    return d

## compute cosine similarity
def cosine_sim(x, y):
    try:
        d = cosine_similarity(x, y)
        d = d[0][0]
    except:
        print(x)
        print(y)
        d = 0.
    return d


############
## TF-IDF ##
############
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(str(doc))) if len(str(doc)) > 0 else ["NA"]


# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r'\w{1,}'
token_pattern = r"\w+"
# token_pattern = r"[\w']+"
tfidf__norm = "l2"
tfidf__max_df = 0.75
tfidf__min_df = 2


def getTFV(token_pattern=token_pattern,
           norm=tfidf__norm,
           max_df=tfidf__max_df,
           min_df=tfidf__min_df,
           ngram_range=(1, 1),
           vocabulary=None,
           stop_words='english'):
    tfv = StemmedTfidfVectorizer(min_df=min_df, max_df=max_df, max_features=None,
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                                 stop_words=stop_words, norm=norm, vocabulary=vocabulary)
    return tfv


#########
## BOW ##
#########
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(str(doc))) if len(str(doc)) > 0 else ["NA"]


# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r'\w{1,}'
token_pattern = r"\w+"
# token_pattern = r"[\w']+"
bow__max_df = 0.75
bow__min_df = 2


def getBOW(token_pattern=token_pattern,
           max_df=bow__max_df,
           min_df=bow__min_df,
           ngram_range=(1, 1),
           vocabulary=None,
           stop_words='english'):
    bow = StemmedCountVectorizer(min_df=min_df, max_df=max_df, max_features=None,
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range,
                                 stop_words=stop_words, vocabulary=vocabulary)
    return bow