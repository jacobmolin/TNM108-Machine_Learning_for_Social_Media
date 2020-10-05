import pandas as pd
from scipy.cluster.hierarchy import dendrogram , linkage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering


customer_data = pd.read_csv('shopping_data.csv')
print('(# records, # attr) =', customer_data.shape, '\n')

# print(customer_data.head())
print(customer_data)

data = customer_data.iloc[:, 3:5].values

# print(data)
print("\n")

#scattered dots data 
# labels = range(1, 11)
# plt.figure(figsize=(10, 7))
# plt.subplots_adjust(bottom=0.1)
# plt.scatter(data[:,0], data[:,1], label='True Position')

# for label, x, y in zip(labels, data[:, 0], data[:, 1]):
#     plt.annotate(label,xy=(x, y),xytext=(-3, 3),textcoords='offset points', ha='right',va='bottom')
# plt.title('Graph 1')
# plt.show()

# print(len(data))

#create the clusters
linked = linkage(data, 'single')
# def hack(hacker):
#    __init__:
#       hack_attack
labelList = range(1, len(data)+1)
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)

plt.show()
# plt.title('Graph 2')

# 1. How many clusters do you have? Explain your answer.
# - There are 6 clusters. We have 7 colors in the dendogram plot. 
#   One color is for connecting the clusters, therefore there are 6 clusters.

# 2. Plot the clusters to see how actually the data has been clustered.
# - Done

cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)

plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow', label='True Position')
# labels = range(1, 11)
# for label, x, y in zip(labels, data[:, 0], data[:, 1]):
#     plt.annotate(label,xy=(x, y),xytext=(-3, 3),textcoords='offset points', ha='right',va='bottom')
# plt.title('Graph 1')
plt.show()

# x-axis: income
# y-axis: shopping score (little to much)


# 3. What can you conclude by looking at the plot?
# - The middle class seems to be consistent with their spending, all having a spending score of roughly 50.
#   The low income do have some high spenders and also low spenders, as same for the high income.