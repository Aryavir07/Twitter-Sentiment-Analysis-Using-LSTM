from flask import Flask, render_template, redirect, request
import pickle,re
import time 
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
app = Flask(__name__)
SENTIMENT_THRESHOLDS = (0.4, 0.7)
SEQUENCE_LENGTH = 200
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"

def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predictNew(modelNew, tokenizerNew,text, include_neutral=True):
    #start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizerNew.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = modelNew.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return { label, float(score)}  


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods = ['GET','POST'])
def tweetSentiment():
    if request.method == 'POST':
        tweet = str(request.form['yourtweet'])
        if len(tweet) == 0:
            return render_template('index.html', yourtweet = 'Enter the tweet first')
        tokenizerNew = pickle.load(open('tokenizer.pkl', 'rb' ))
        encoder = pickle.load(open('encoder.pkl','rb' ))
        modelNew = keras.models.load_model('model.h5')
        thistuple = [score, sentiment]  = predictNew(modelNew, tokenizerNew, tweet)
        
        print(type(sentiment), type(score))
        sentiment = ''
        for x in thistuple:
            if type(x) == str:
                sentiment = x
        finaltweet = "Your tweet " +"' "+ tweet +" '"+" is a " + sentiment[0].upper() + sentiment[1:].lower() + " tweet " + "with score "  + str(score) + "."
    return render_template('index.html', yourtweet = finaltweet)


if __name__ == '__main__':
    app.run(debug = False)