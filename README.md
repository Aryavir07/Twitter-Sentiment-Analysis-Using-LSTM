# Twitter-Sentiment-Analysis-Using-LSTM and Gensim
In this notebook, I have implemented Stacked LSTM with embedding to analyse 1.6Million tweets which is divided into three categories 1. Positive 2. Negative 3. Neutral, made model to predict class of new tweets with accuracy of 78 percent.



## Final results after deployment
![2](https://user-images.githubusercontent.com/42632417/111740892-88276380-88ab-11eb-80ba-14bea14bbf64.png)


## Performance
|       | precision | recall | f1-score | support |
| :--: | :--:      | :--:   | :--:     |  :--:   |
|0      | 0.78    |  0.75   |   0.76  |   79800 |
|1      | 0.76    |  0.79   |   0.77   |  80200 |
|accuracy  |        |        |   0.77  |  160000 |
 |macro avg   |    0.77  |    0.77   |   0.77  |  160000 |
 |weighted avg  |     0.77   |   0.77   |   0.77  |  160000 |
 
 
 
 ![__results___71_0](https://user-images.githubusercontent.com/42632417/110976520-fb4e4880-8386-11eb-9c1d-2e9dba59fc6c.png) ![__results___71_1](https://user-images.githubusercontent.com/42632417/110976528-ff7a6600-8386-11eb-85de-27c020f8c486.png)
 
 # Using Gensim
 ![download](https://user-images.githubusercontent.com/42632417/111896580-b8ecd180-8a40-11eb-9df6-4f4f453fa66a.png)



## Model

![__results___67_0](https://user-images.githubusercontent.com/42632417/110977050-abbc4c80-8387-11eb-98c2-8fd62539f630.png)


## Datasets
- Stanford's GloVe 100d word embeddings : https://www.kaggle.com/danielwillgeorge/glove6b100dtxt/tasks
- Sentiment140 dataset with 1.6 million tweets : https://www.kaggle.com/kazanova/sentiment140

## Paper :
https://www.academia.edu/35947062/Twitter_Sentiment_Analysis_using_combined_LSTM_CNN_Models
