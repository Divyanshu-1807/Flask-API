from flask import Flask,request
import pickle
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import MultinomialNB
import pandas as pd

spam=pickle.load(open('spam.pkl','rb'))
data=pd.DataFrame(spam)

tfid=TfidfVectorizer(max_features=3000)

tfid.fit_transform(data['new_text'])
#Y=predict['target'].values
#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
#mnb=MultinomialNB()
#mnb.fit(X_train,Y_train)

model=pickle.load(open('model.pkl','rb'))

list=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

ps=PorterStemmer()
def action(text):
    text=text.lower()
    text=text.split()

    y=[]
    for words in text:
      if words.isalnum():
        y.append(words)

    text=y[:]
    y.clear()
    for words in text:
        if words not in list and words not in string.punctuation:
          y.append(words)

    text=y[:]
    y.clear()

    for words in text:
        y.append(ps.stem(words))

    return " ".join(y)

app=Flask(__name__)

@app.route('/')
def home():
    return "HELLO WORLD"

@app.route('/predict',methods=['POST'])
def predict():
    text=request.get_json()
    input_text=action(text['text'])
    final_text=tfid.transform([input_text])
    result=model.predict(final_text)[0]
    return {'spam':str(result)}

if __name__ == '__main__':
    app.run(debug=True)
