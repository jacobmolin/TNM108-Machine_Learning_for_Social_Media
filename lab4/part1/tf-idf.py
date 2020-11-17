from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.metrics.pairwise import cosine_similarity
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

d0 = "The sky is blue."
d1 = "The sun is bright."
d2 = "The sun in the sky is bright."
d3 = "We can see the shining sun, the bright sun."
Z = (d0, d1, d2, d3)

vectorizer = CountVectorizer()

my_stop_words = {"the", "is"}
my_vocabulary = {'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}
vectorizer = CountVectorizer(
    stop_words=my_stop_words, vocabulary=my_vocabulary)

# print(vectorizer.vocabulary)
# print(vectorizer.stop_words)

smatrix = vectorizer.transform(Z)
# print(smatrix)

matrix = smatrix.todense()
# print(matrix)


# -------- Computing the tf-idf score ------------

tfidf_transformer = TfidfTransformer(norm="l2")
tfidf_transformer.fit(smatrix)

# print idf values
feature_names = vectorizer.get_feature_names()
df_idf = pd.DataFrame(tfidf_transformer.idf_,
                      index=feature_names, columns=["idf_weights"])
# sort ascending
df_idf.sort_values(by=['idf_weights'])
# print(df_idf, '\n')

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(smatrix)

# get tfidf vector for first document
first_document = tf_idf_vector[0]  # first document "The sky is blue."

# print the scores
df = pd.DataFrame(first_document.T.todense(),
                  index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"], ascending=False)
# print(df, '\n')


# -------- Document Similarity ---------------

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
# (4,11) = (documents, unique words across all docs)
# print(tfidf_matrix.shape, '\n')

cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
# print(type(cos_similarity))
# print(cos_similarity, '\n')

# Take the cos similarity of the third document (cos similarity=0.52)
angle_in_radians = math.acos(cos_similarity[0][2])
# print(math.degrees(angle_in_radians))


# -------- Classifying Text ---------------

data_train = load_files(
    '/Users/jacobmolin/Dropbox/LiU/Åk4/HT1/TNM108/Labs/lab4/20news-bydate/20news-bydate-train')
# data = fetch_20newsgroups()
# data_train = fetch_20newsgroups(subset='train')
# data_test = fetch_20newsgroups(subset='test')
# data_train = load_files('/Users/jacobmolin/Dropbox/LiU/Åk4/HT1/TNM108/Labs/lab4/20news-bydate/20news-bydate-train')

# print(data_train.target_names)

my_categories = ['rec.sport.baseball',
                 'rec.motorcycles', 'sci.space', 'comp.graphics']
# train = fetch_20newsgroups(subset='train', categories=my_categories)
train = load_files('/Users/jacobmolin/Dropbox/LiU/Åk4/HT1/TNM108/Labs/lab4/20news-bydate/20news-bydate-train',
                   categories=my_categories)

# test = fetch_20newsgroups(subset='test', categories=my_categories)
test = load_files('/Users/jacobmolin/Dropbox/LiU/Åk4/HT1/TNM108/Labs/lab4/20news-bydate/20news-bydate-test',
                  categories=my_categories)

# print(len(train.data))
# print(len(test.data))
# print(train.data[9])


cv = CountVectorizer(encoding='latin-1')
# print(type(train.data))
X_train_counts = cv.fit_transform(train.data)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

model = MultinomialNB().fit(X_train_tfidf, train.target)

docs_new = ['Pierangelo is a really good baseball player', 'Maria rides her motorcycle',
            'OpenGL on the GPU is fast',
            'Pierangelo rides his motorcycle and goes to play football since he is a good football player too.']
X_new_counts = cv.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = model.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, train.target_names[category]))
