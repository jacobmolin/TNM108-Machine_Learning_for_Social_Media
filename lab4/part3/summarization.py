from summa.summarizer import summarize
from summa import keywords

text = "At an age when most top programmers are enjoying their retirement, 64-year old Guido van Rossum decided \"retirement was boring\" and came back to work for -- drumroll please -- Microsoft. Van Rossum is best known as the creator of the popular open-source language Python. To say this move caught people by surprise is a vast understatement. Microsoft, for one, though is delighted by his move. A Microsoft spokesperson said, \"We're excited to have him as part of the Developer Division. Microsoft is committed to contributing to and growing with the Python community, and Guido's on-boarding is a reflection of that commitment.\" Also: Programming language Python's popularity: Ahead of Java for first time but still trailing C They should be. Van Rossum, who created Python in 1989 and became its Benevolent Dictator for Life (BDFL), is widely respected as one of open-source's greatest programmers. Python is one of the world's most widely used languages. It's one of the foundation languages of the globe's most important software stack: Linux, Apache, MySQL, Python/Perl/PHP (LAMP). Thanks to its use in Machine Learning (ML), Python is showing no signs of slowing down. While van Rossum stepped down as Python's BDFL in 2018, he's remained active in Python development circles. He is also still the president of the Python Software Foundation. This group oversees the Python language. Over the years, van Rossum has worked for many companies. This has included Zope, a Python-based web application server organization; Google; and the personal cloud storage company Dropbox which is built on Python. No matter the company; no matter the job title; van Rossum kept working on improving Python. We can be sure he'll continue to do so at Microsoft. For years, Microsoft took little active interest in Python thanks to a \"Not invented here\" attitude. As Microsoft started working more with open source and the cloud, the company changed its stance. As Steve Dower, a Microsoft software engineer explained, Microsoft came around to working with Python first with Python Tools for Visual Studio(PTVS) in 2010 and then IronPython, which runs on .NET.  By 2018, \"we are out and proud about Python, supporting it in our developer tools such as Visual Studio and Visual Studio Code, hosting it in Azure Notebooks, and using it to build end-user experiences like the Azure CLI.\" In short, Python is \"one of the essential languages for services and teams to support, as well as the most popular choice for the rapidly growing field of data science and analytics both inside and outside of the company, \" concluded Dower. That's truer today than ever. Microsoft hiring van Rossum is one of the smartest moves it could make to solidify it both as a leading software development company and a true open-source believer."
# text = "TextRank is an unsupervised algorithm for the automated summarization of texts that can also be used to obtain the most important keywords in a document. The algorithm applies a variation of PageRank over a graph constructed specifically for the task of summarization. This produces a ranking of the elements in the graph: the most important elements are the ones that better describe the text. This approach allows TextRank to build summaries without the need of a training corpus or labeling and allows the use of the algorithm with different languages. TextRank models any document as a graph using sentences as nodes. A function to compute the similarity of sentences is needed to build edges in between. This function is used to weight the graph edges, the higher the similarity between sentences the more important the edge between them will be in the graph. In the domain of a Random Walker, as used frequently in PageRank, we can say that we are more likely to go from one sentence to another if they are very similar."

# Define length of the summary as a proportion of the text
print('\n Summarization, ratio=0.2:')
print(summarize(text, ratio=0.2))

print('\n Summarization, 50 words:')
print(summarize(text, words=50), '\n')

print('Summarization, 100 words:')
print(summarize(text, words=100), '\n')

print("Keywords:\n", keywords.keywords(text), '\n')


with open('./part3/venezuela.txt', 'r') as file:
    data = file.read().replace('\n', '')

print('Summarization Venezuela, 200 words:')
print(summarize(data, words=200), '\n')

# to print the top 3 keywords
# print("Top 3 Keywords:\n", keywords.keywords(text, words=3), '\n')
