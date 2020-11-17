# Answers to Lab4

## Part 1: Text feature extraction tf-idf

1. What is the TF-IDF measure (term frequency – inverse document frequency)?

It calculates how important a word is to a document in a collection. It does so by scaling up rare terms while scaling down frequent terms. A high TF-IDF score is achieved for a word that have a high frequency in one document, but low frequency across the whole collection of documents.

2. How to use TF-IDF for document similarity and classify text?

For document similarity the angle between two document vectors is calculated. The direction of the document vector is what is important, not the magnitude (i.e. the length of the document). cos(angle) is what is used to compare document vectors. cos(angle) can be solved from the dot product equation for cos(angle). By looking at the angle the cosine similarity will produce a metric (0-1) that says how related two documents are.

A small angle gives a high document similarity. A nearly orthogonal angle means that the documents are unrelated. Angles around 180 deg. means opposite document similarity.

For classifying text a common method is to use the multinomial naive Bayes. A predictive model can be created by first calculating the TF-IDF matrix for a text collection and then passing it to a multinomial naive Bayes classifier to train the model.

## Part 2: Text sentiment analysis

make nice pipeline

## Part 3: Text summarization

1. Explain the TextRank algorithm and how it works to the lab assistant.

TextRank creates a graph where the text's sentences are the nodes.
The similarity between two nodes is calculated to create the edges. The higher the similarity between the nodes (sentences), the higher the weight on the edges. TextRank calculates the similarity by comparing the lexical tokens of the sentences and divides by the length of both sentences in consideration.

The result is a dense graph. TextRank then uses PageRank to compute the importance of each node in the dense graph. The most important sentences are selected and are returned in the order they appear in the original text, i.e. a summary.

2. Show your lab assistant some summaries you created; and discuss the quality of the summaries (like; does your abstract make any sense? can you create a summary that looks like an abstract from a news article? can you summarize product opinions from customers etc).

Summarize 1: From an article about Microsoft hiring Python creator Van Rossum. The summarize is an accurate representation of the article.

Summarize 2: The text below "TextRank" from the laboration description. The text seems to be too short to be summarized.

TextRank is an unsupervised algorithm for the automated summarization of texts that can also be used to obtain the most important keywords in a document.
The algorithm applies a variation of PageRank over a graph constructed specifically for the task of summarization.
TextRank models any document as a graph using sentences as nodes.
A function to compute the similarity of sentences is needed to build edges in between.
This function is used to weight the graph edges, the higher the similarity between sentences the more important the edge between them will be in the graph.
