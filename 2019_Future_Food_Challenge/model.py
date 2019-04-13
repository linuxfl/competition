import sys
import warnings
import numpy as np
import pandas as pd 
import json
import re
import matplotlib.pyplot as plt
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

pd.set_option('display.max_colwidth', 500)
if not sys.warnoptions:
    warnings.simplefilter("ignore")

df_train = pd.read_table('[Updated] Training Set for Competition.txt', sep='\t', encoding='utf-8')
df_test = pd.read_table('[Updated] Test Set for Competition.txt', sep='\t', encoding='utf-8')

df_train['Date'] = pd.to_datetime(df_train['Date'], format='%d.%m.%Y').dt.date
df_train['Event ID'] = df_train['Event ID'].fillna(0)
df_train.head(3)

pattern = re.compile(r'^[a-zA-Z]+$')
def filter_token(token):
    token = token.lower()
    if token in stop_words:
        return False
    if not pattern.findall(token):
        return False
    if token == "https" or token == "http":
        return False
    return True
    
def clean(text):
    tokens = nltk.word_tokenize(text)
    return [token.lower() for token in tokens if filter_token(token)]

df_train.dropna(subset=['Abstract'], inplace=True)
df_train['ab_clean'] = df_train['Abstract'].apply(clean)
df_train[['Abstract','ab_clean']].head(3)

bigram_model = Phraser(Phrases(df_train['ab_clean']))
df_train['ab_clean_ph'] = df_train['ab_clean'].apply(lambda text: bigram_model[text])
df_train[['Abstract','ab_clean','ab_clean_ph']].head(3)

vectorizer = TfidfVectorizer(min_df=2, max_features=1000)
text_tfidf = vectorizer.fit_transform([' '.join(text) for text in df_train['ab_clean_ph']])
X_train = text_tfidf
y_train = df_train['Event ID']

param_test_lr =  {'C' :[10 ** x for x in list(range(-4,5,1))]}
param_test_lr['C']
gsearch_lr = GridSearchCV(estimator = LogisticRegression( 
        penalty='l1', 
        solver='liblinear',
        max_iter=1000,
        class_weight = 'balanced'
    ),                   
    param_grid = param_test_lr, 
    cv = 5,
    n_jobs = 4)

gsearch_lr.fit(X_train,y_train)

GridSearchCV(cv=5, error_score='raise',
       estimator=LogisticRegression(C=1.0, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=1000,
          multi_class='ovr', n_jobs=1, penalty='l1', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
       fit_params=None, iid=True, n_jobs=4,
       param_grid={'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)

print gsearch_lr.best_params_
print gsearch_lr.best_score_
