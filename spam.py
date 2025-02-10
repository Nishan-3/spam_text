import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import nltk 
import nltk
nltk.download('punkt','stopwords')

from nltk.corpus import stopwords

import string
from nltk.stem.porter import PorterStemmer 

df = pd.read_csv('spam.csv',encoding='latin1')

print(df.shape)
#data cleaning . 
#lets check  if all datas are important or not . 
#here the last 3 coloms are  really not helpful they dont have any values . 

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

#re naming the columns  
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
#print(df.sample(5))

#EDA
"""
here we have either ham or spam, the spam is not here but we have inout data set so here we want to use the label encoder and make it either 1 or 0 
     target                                               text
1632    ham  Hello my little party animal! I just thought I...
4604    ham  I need an 8th but I'm off campus atm, could I ...
302     ham  Oh and by the way you do have more food in you...
4032    ham  I am taking you for italian food. How about a ...
1230    ham  I want to send something that can sell fast.  ...
"""
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target']=encoder.fit_transform(df['target'])
#print(df.duplicated().sum())
df= df.drop_duplicates(keep='first')
#print(df.duplicated().sum())
#we have to make 3 columns .
df['num_characters'] = df['text'].apply(len)
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sentence'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

#description for the target #spam
ham_df =df[df['target'] ==0][['num_characters','num_words','num_sentence']].describe()

spam_df =df[df['target'] ==1][['num_characters','num_words','num_sentence']].describe()


#Data Pre processing (Lower case , Tokenization,Removing Special characters,removing words and puncuatuations , stemming .
#    )

def transform_text(text):
    ps= PorterStemmer()
    text = text.lower()
    text  = nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:] #list cloning 
    y.clear #here we cleared the y  becasue we have to check the english word and string puntuation . 
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
        text = y[:]
        y.clear()
        #steming
    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)  

#applying the transformation function that we made in the text coluns 
df['transformed_text']= df['text'].apply(transform_text)

#here we are splitting the  words of spam 
spam_corpus =[]
for msg in df[df[df['target']] ==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

#here we are spliting the words of  ham 
ham_corpus =[]
for msg in df[df[df['target']] ==0]['transformed_text'].tolist():
    for word in msg.split():
       ham_corpus.append(word)

#here we will try naive bayes (Model builiding ) #textual data here naive bayes best gives 
#we have to convert the text into the vectorize form . 
#here we have bag  of words , tfidf , wordtovec . ,here we have bags of words . 
from sklearn.feature_extraction.text  import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tf = TfidfVectorizer(max_features=3000)
x=cv.fit_transform(df['transformed_text']).toarray()
#using the scaler 
'''from sklearn.preprocessing import MinMaxScaler #the reason that we are using the MinMax scaler is that standard scaler give  us value between the - and + and our naive bayes doesnot accept that
scaler = MinMaxScaler()
x=scaler.fit_transform(x) we are commenting out because we didint see any good performance here  using the scaler '''

y=df['target'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB 
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
BN= BernoulliNB()
BN.fit(x_train,y_train)
y_pred = BN.predict(x_test)

