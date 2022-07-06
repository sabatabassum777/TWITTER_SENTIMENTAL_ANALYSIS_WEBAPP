import nltk #Natural Language Tool Kit
nltk.download(['punkt', 'wordnet'])
nltk.download('omw-1.4')

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def tokenize(text):

    tokens = word_tokenize(text)
    #print("By doing Tokenization:")
    #print(tokens)

    lemmatizer = WordNetLemmatizer() #converts the words(tokens) into their base form wrto the context by mapping with WordNet Cloud

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
   # print("After Performing Lemmatization:")
    return clean_tokens

path= "Tweets.csv"
def load_data(path):
    df=pd.read_csv(path)
    #cleaning the data
    df.dropna(inplace=True)
    df=df.drop(['textID','text'],axis=1)
    #Label Encoding
    from sklearn import preprocessing
    LE=preprocessing.LabelEncoder()
    df.sentiment=LE.fit_transform(df.sentiment)
    #labelling
    negative=df[df.sentiment==0]
    neutral=df[df.sentiment==1]
    positive=df[df.sentiment==2]
    #loading the data
    x=df.selected_text.values
    y=df.sentiment.values
    return x,y

vect=CountVectorizer(tokenizer=tokenize)

url="Tweets.csv"
x,y=load_data(url)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0,shuffle=True)
#print(x_train)
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
vect=CountVectorizer(tokenizer=tokenize)# (lemmatization)generates frequency table for tokens for all documents
tfidf=TfidfTransformer()#highlights the words  which are import in the comment
from sklearn.ensemble import RandomForestClassifier
random=RandomForestClassifier()
#train our model
#train classifier
x_train_count=vect.fit_transform(x_train)
x_train_tfidf=tfidf.fit_transform(x_train_count)
random.fit(x_train_tfidf,y_train)

#predict on test data
x_test_count=vect.transform(x_test)
x_test_tfidf=tfidf.transform(x_test_count)
y_pred=random.predict(x_test_tfidf)

print("Test Accuracy")
accuracy=random.score(x_test_tfidf,y_test)*100
print(accuracy)
print("Overall Accuracy")
x1=vect.transform(x)
x1_tfidf=tfidf.transform(x1)
print(random.score(x1_tfidf,y)*100)

import pickle
pickle.dump(random,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print("Success loaded")
