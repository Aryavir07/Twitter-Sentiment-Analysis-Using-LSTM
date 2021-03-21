import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
Twitter Sentiment Analysis with LSTM

# In[68]:


df = pd.read_csv('./dataset/tweets.csv', header = None, )


# In[69]:


df.shape





df.head(5)


# - 0 - target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
# - 1 - id of user 
# - 2 - date of tweer 
# - 3 - unnecessary column 
# - 4 - nickname of author 
# - 5 - tweet content |

df = df.rename(columns={0: 'target', 1: 'id', 2: 'date', 3: 'query', 4: 'username', 5: 'content'})
df.head()

df.info()
df.isnull().sum() # no missing values :)
missing_data = df.isnull().sum().sort_values(ascending=False)
percentage_missing = round((df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100,2)
missing_info = pd.concat([missing_data,percentage_missing],keys=['Missing values','Percentage'],axis=1)
missing_info

pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 100)

negativeTweets = df[df['target']==0]['content'].count()
print(negativeTweets)
neutralTweets = df[df['target']==2]['content'].count()
print(neutralTweets)
positiveTweets = df[df['target']==4]['content'].count()
print(positiveTweets)
df[df['target']==0]['content'].head()
df[df['target']==2]['content'].head()
df[df['target']==4]['content'].head()

# there are negative tweets from 0 to 800000 and the from 800000 to 160000 positive tweets 
# I shuffled it randomly
# also there is no neutral tweets
from sklearn.utils import shuffle
df = shuffle(df)

df['target'] = df['target'].replace([0,4],['Negative','Positive'])

fig = plt.figure(figsize=(5,5))
targets = df.groupby('target').size()
targets.plot(kind='pie', subplots=True, figsize=(10, 8), autopct = "%.2f%%", colors=['red','green'])
plt.title("Pie chart of different classes of tweets",fontsize=16)
plt.ylabel("")
plt.legend()
plt.show()

df['target'].value_counts()

# length of tweets in both the classes
#df[df['target']==2]['content'].head()
df['length'] = df.content.str.split().apply(len)


# Describing length of tweets in the positive class
df[df['target']=='Positive']['length'].describe().round(3)

# Describing length of tweets in the negative class
df[df['target']=='Negative']['length'].describe().round(3)


# In[89]:


fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(121)
sns.distplot(df[df['target']=='Positive']['length'], ax=ax1,color='blue')
describe = df.length[df.target=='Positive'].describe().to_frame().round(2)

ax1 = fig.add_subplot(122)
sns.distplot(df[df['target']=='Negative']['length'], ax=ax1,color='black')
describe = df.length[df.target=='Negative'].describe().to_frame().round(2)
fig.suptitle('Distribution of text length for positive and negative sentiment tweets.', fontsize=10)

plt.show()


# ### Top 10 Users with Maximum Number of Tweets

# In[90]:


plt.figure(figsize=(14,7))
common_keyword=sns.barplot(x=df[df['target']=='Positive']['username'].value_counts()[:10].index,
                           y=df[df['target']=='Positive']['username'].value_counts()[:10],palette='magma')
common_keyword.set_xticklabels(common_keyword.get_xticklabels(),rotation=90)
common_keyword.set_ylabel('Positive tweet frequency',fontsize=12)
plt.title('Top 10 users who publish positive tweets',fontsize=16)
plt.show()


# In[91]:


df[df['username']=='what_bugs_u'].head(4)


# In[92]:


df[df['username']=='DarkPiano'].head(4)


# In[93]:


plt.figure(figsize=(14,7))
common_keyword=sns.barplot(x=df[df['target']=='Negative']['username'].value_counts()[:10].index,
                           y=df[df['target']=='Negative']['username'].value_counts()[:10],palette='magma')
common_keyword.set_xticklabels(common_keyword.get_xticklabels(),rotation=90)
common_keyword.set_ylabel('Positive tweet frequency',fontsize=12)
plt.title('Top 10 users who publish positive tweets',fontsize=16)
plt.show()
df[df['username']=='lost_dog']['content'].head(4)


from wordcloud import WordCloud, STOPWORDS


# https://www.pluralsight.com/guides/natural-language-processing-visualizing-text-data-using-word-cloud

plt.figure(figsize=(12,6))
word_cloud = WordCloud(stopwords = STOPWORDS, max_words = 200, width=1366, height=768, background_color="white").generate(" ".join(df[df.target=='Positive'].content))
plt.imshow(word_cloud,interpolation='bilinear')
plt.axis('off')
plt.title('Most common words in positive sentiment tweets.',fontsize=10)
plt.show()

plt.figure(figsize=(12,6))
word_cloud = WordCloud(stopwords = STOPWORDS, max_words = 200, width=1366, height=768, background_color="white").generate(" ".join(df[df.target=='Negative'].content))
plt.imshow(word_cloud,interpolation='bilinear')
plt.axis('off')
plt.title('Most common words in positive sentiment tweets.',fontsize=10)
plt.show()

# dropping columns which are not usefull in analysis

df.drop(['id','date','query','username','length'], axis=1, inplace=True)
df.head(4)

df.target = df.target.replace({'Positive':1, 'Negative':0})

df.head()


# ### Cleaning tweets
# - Stopword removal
# - Stemming 

# In[100]:


from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')


# #### stemming means => running , runner, runned to : run
# #### Which stemmer to choose ?
# https://stackoverflow.com/questions/10554052/what-are-the-major-differences-and-benefits-of-porter-and-lancaster-stemming-alg



english_stopwords = stopwords.words('english')
stemmer = SnowballStemmer('english')
regex = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+" #regex for mentions and links in tweets


def preprocess(content, stem=False):
  content = re.sub(regex, ' ', str(content).lower()).strip()
  tokens = []
  for token in content.split():
    if token not in english_stopwords:
      tokens.append(stemmer.stem(token))
  return " ".join(tokens)



df.content = df.content.apply(lambda x: preprocess(x))


df.head()


# ### Splitting Data Into train and test 

train, test =  train_test_split(df, test_size = 0.1, random_state = 4)

print('train dataset size: {}'.format(train.shape))
print('test dataset size: {}'.format(test.shape))


# ### Tokenization
# 1. fit_on_texts Updates internal vocabulary based on a list of texts. This method creates the vocabulary index based on word frequency. So if you give it something like, "The cat sat on the mat." It will create a dictionary s.t. word_index["the"] = 1; word_index["cat"] = 2 it is word -> index dictionary so every word gets a unique integer value. 0 is reserved for padding. So lower integer means more frequent word (often the first few are stop words because they appear a lot).
# 
# 2. texts_to_sequences Transforms each text in texts to a sequence of integers. So it basically takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary. Nothing more, nothing less, certainly no magic involved.
# 
# 3.Why don't combine them? Because you almost always fit once and convert to sequences many times. You will fit on your training corpus once and use that exact same word_index dictionary at train / eval / testing / prediction time to convert actual text into sequences to feed them to the network. So it makes sense to keep those methods separate.

# 
# Lets see what this line of code does.
# 
# tokenizer.fit_on_texts(text)
# 
# For example, consider the sentence " The earth is an awesome place live"
# 
# tokenizer.fit_on_texts("The earth is an awesome place live") fits [[1,2,3,4,5,6,7]] where 3 -> "is" , 6 -> "place", so on.
# 
# sequences = tokenizer.texts_to_sequences("The earth is an great place live")
# returns [[1,2,3,4,6,7]].
# 


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train.content)
vocab_size = len(tokenizer.word_index) + 1
max_length = 65


train_sequences = tokenizer.texts_to_sequences(train.content)
test_sequences = tokenizer.texts_to_sequences(test.content)


# #### pad sequence 
# 
# >> pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]])
# 
# output: 
# >>>        array([[0, 1, 2, 3],
#                [3, 4, 5, 6],
#                [0, 0, 7, 8]], dtype=int32)

X_train =  pad_sequences(train_sequences, maxlen = max_length, padding = 'post')
X_test = pad_sequences(test_sequences, maxlen = max_length, padding = 'post')

Y_train = train.target.values # np array
Y_test = test.target.values

Y_train[0:10]


# ## Word embedding Using Glove

# - we need to load the entire GloVe word embedding file into memory as a dictionary of word to embedding array.


embedding_index = dict()
f = open('glove.6B.100d.txt', encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embedding_index[word] = coefs
f.close()
print('Loaded %s words in vectors.'%len(embedding_index))


#create a weight matrix for words in training docs
embeddings_matrix = np.zeros((vocab_size, 100)) # 100 is embedding dimension
for word, index in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[index] = embedding_vector

embedding_layer = tf.keras.layers.Embedding(vocab_size, 100, input_length=max_length, weights=[embeddings_matrix], trainable=False)

model = Sequential([
        embedding_layer,
        tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(LSTM(128)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])



model.summary()
import pydot
import graphviz
tf.keras.utils.plot_model(model, show_shapes=True)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size = 1000, epochs=2, validation_data=(X_test, Y_test), verbose=2)
y_pred = model.predict(X_test)
y_pred = np.where(y_pred>0.5, 1, 0)

print(classification_report(y_test, y_pred))



#History for accuracy
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train accuracy', 'Test accuracy'], loc='lower right')
plt.show()
# History for loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train loss', 'Test loss'], loc='upper right')
plt.suptitle('Accuracy and loss for second model')
plt.show()




