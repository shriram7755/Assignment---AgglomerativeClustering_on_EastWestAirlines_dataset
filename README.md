# Assignment---AgglomerativeClustering_on_EastWestAirlines_dataset
Dataset : EastWestAirlines

Agglomerative clustering is a hierarchical clustering algorithm that builds a tree of clusters. The algorithm starts by treating each data point as a single cluster and then iteratively merges the closest pairs of clusters until only one cluster remains. This hierarchical structure can be represented as a dendrogram.

Here's a brief overview of how agglomerative clustering works:

1)Initialization: Treat each data point as a single cluster.

2)Compute pairwise distances: Calculate the distance (or similarity) between each pair of clusters. Various distance metrics can be used, such as Euclidean distance, Manhattan distance, or others, depending on the nature of the data.

3)Merge the closest clusters: Find the two clusters with the smallest distance and merge them into a single cluster. This process is repeated until only one cluster remains.

4)Update distances: Recalculate the distances between the new cluster and all remaining clusters.

5)Repeat: Steps 3 and 4 are repeated until only one cluster remains, containing all the data points.

The choice of distance metric and linkage criteria (the method for determining the distance between clusters) can significantly impact the results of agglomerative clustering. Common linkage criteria include:

Single linkage: The distance between two clusters is defined as the smallest distance between any two points in the clusters.

Complete linkage: The distance between two clusters is defined as the largest distance between any two points in the clusters.

Average linkage: The distance between two clusters is defined as the average distance between all pairs of points from the two clusters.

Ward's linkage: Minimizes the variance within the clusters being merged.

Agglomerative clustering is a partitional clustering method, meaning it divides the data into non-overlapping subsets or clusters. It is widely used in various fields, including biology, image analysis, and document clustering. One advantage of agglomerative clustering is that it does not require specifying the number of clusters beforehand, as the dendrogram provides a hierarchical view that allows users to choose the number of clusters based on their needs.
