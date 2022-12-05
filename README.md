# sentiment-analysis-leveraging-lstm

# Sentiment Analysis of Product Reviews leveraging LSTM Network

`Sentiment Analysis` is determining whether a written piece of text has a `positive`, `neutral`, or `negative`
connotation. These written pieces of text are usually the reviews that are left by customers once they use products,
brands, services, and so forth. These reviews give an insight into how appealing or off-putting a particular product,
brand, or service was to the customer. These `insights` are extremely useful because they are not only an indicator of
`customer satisfaction` but also companies can use them to `drive business decisions`.

Three different sentiment analysis model are built leveraging a `deep learning` approach that will utilize the `customer
reviews` of `Amazon products`. Since `Long Short Term Memory Network` (LSTM) is very effective in dealing with long sequence
data and learning long-term dependencies, it is used for automatic sentiment classification of future product reviews.

![](https://github.com/hardikasnani/sentiment-analysis-leveraging-lstm/blob/main/screenshot/people-sentiment.png)

## Table of Contents

- [Highlights](#Highlights)
- [Dataset](#Dataset)
- [Approach](#Approach)
- [Screenshot](#Screenshot)
- [Information About Folders and Files](#Information-About-Folders-and-Files)
- [License](#License)
- [References](#References)
  - [Papers](#Papers)
  - [Links and Blogs](#Links-and-Blogs)
- [End Notes](#End-Notes)

## Highlights

Following are the highlights of the project:
- Sentiment Analysis of Amazon Product Reviews using an `imbalanced dataset`
- The initial sentiment model is trained and evaluated using the following sentiment distribution:
  - `Positive Reviews`: `89.02%`
  - `Neutral Reviews`: `5.09%`
  - `Negative Reviews`: `5.71%`
- Usage of pre-trained `GloVe Word Embeddings`
- Explored `different settings` to build the sentiment model based on the following:
  - Batch Size
  - Number of LSTM Layers
  - Number of Units per LSTM Layer
  - Dropout Values
  - Absence or Presence of Dense Layer before the output layer
  - Epochs
  - Patience during Early Stopping
  - Word Stemming or Lemmatizing
- Trained and Evaluated additional sentiment models by addressing the imbalance in data using the following methods:
  - Assigned `class weights` during the model training
  - Used `SMOTE` to `synthetically` create the `oversampled data`
- `Comparison` between different sentiment models

![](https://github.com/hardikasnani/sentiment-analysis-leveraging-lstm/blob/main/screenshot/sentiment-distribution.png)

## Dataset

Consumer Reviews of Amazon Products is the dataset that will be used. It has a reasonable dimension i.e. it has over
34,000 consumer reviews for Amazon products like the Kindle, Fire TV Stick, and so forth. The dataset includes basic
product information such as name, review title, review text, review rating, and more for each product. The dataset is
publicly available on [Kaggle](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products).

In this dataset, the column reviews.rating has values ranging from 1 to 5. These values will be updated so that each of
them corresponds to a sentiment. Values 1 and 2 will be treated as a negative sentiment, value 3 will be treated as a
neutral sentiment, and values 4 and 5 will be treated as a positive sentiment.

## Approach

Exploratory Dataset Analysis is done for the above-mentioned dataset. The text column is cleaned and the data
is then split into training, testing, and validation sets. This data is then tokenized and padded followed by preparing
the word embedding that helps in setting up the embedding layer for the sentiment model. Evaluation Metric is finalized
and different settings are explored to build the sentiment model. Apart from the initial model that is trained and
evaluated using the imbalanced data, two other models are built. One of the models is trained using class weights and
the other model is trained using synthetically oversampled data. Finally, the results are compared for different models
trained and evaluated under the best setting.

## Information About Folders and Files

- `dataset/1429_1.csv`: Dataset of 34,660 consumer reviews for Amazon products
- `dataset/additional_dataset.txt`: Provides links to additional Dataset of 5,000 + 28,000 consumer reviews for Amazon
- `screenshot/people-sentiment.png`: Screenshot of the people with negative, neutral, and positive facial expressions
- `screenshot/sentiment-distribution.png`: Screenshot of the imbalanced dataset
- `sentiment-analysis-lstm.ipynb`: Google Colab notebook for the project

## License

This project is licensed under the MIT License and for more details, see the [LICENSE.md](https://github.com/hardikasnani/sentiment-analysis-leveraging-lstm/blob/main/LICENSE) file

## References

Here are some references I looked at while working on this project:

### Papers

- K. Baktha and B. K. Tripathy, "Investigation of recurrent neural networks in the field of sentiment analysis," 2017
International Conference on Communication and Signal Processing (ICCSP), 2017, pp. 2047-2050,
doi:10.1109/ICCSP.2017.8286763.
- T. Kati ́c and N. Mili ́cevi ́c, ”Comparing Senti- ment Analysis and Document Representation Meth- ods of Amazon Reviews,”
2018 IEEE 16th Inter- national Symposium on Intelligent Systems and In- formatics (SISY), 2018, pp. 000283-000286,
doi: 10.1109/SISY.2018.8524814.
- J. C. Gope, T. Tabassum, M. M. Mabrur, K. Yu and M. Arifuzzaman, ”Sentiment Analysis of Ama- zon Product Reviews Using
Machine Learning and Deep Learning Models,” 2022 International Con- ference on Advancement in Electrical and Electronic
Engineering (ICAEEE), 2022, pp. 1-6, doi: 10.1109/ICAEEE54957.2022.9836420.
- N. Sharm, T. Jain, S. S. Narayan and A. C. Kan- dakar, ”Sentiment Analysis of Amazon Smartphone Reviews Using Machine
Learning Deep Learning,” 2022 IEEE International Conference on Data Science and Information System (ICDSIS), 2022, pp.
1-4, doi: 10.1109/ICDSIS55133.2022.9915917.

### Links and Blogs

- https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17
- https://www.kaggle.com/code/lystdo/lstm-with-word2vec-embeddings/script
- https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010
- https://www.tensorflow.org/api_docs/python/tf/keras
- https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
- https://towardsdatascience.com/smote-fdce2f605729
- https://medium.com/geekculture/10-hyperparameters-to-keep-an-eye-on-for-your-lstm-model-and-other-tips-f0ff5b63fcd4
- https://medium.com/@saumya.ranjan/how-to-write-a-readme-md-file-markdown-file-20cb7cbcd6f
- https://github.com/banesullivan/README

## End Notes

Did you find this project useful? Which other setting do you think can be explored? In which other way can the imbalance
in this data be handled? Feel free to discuss your experiences on the [discussion portal](https://github.com/hardikasnani/sentiment-analysis-leveraging-lstm/discussions),
and I'll be more than happy to discuss.

[Back to Top](#Table-of-Contents)

