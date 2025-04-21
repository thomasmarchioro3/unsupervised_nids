import numpy as np

from scipy.cluster.hierarchy import linkage, to_tree, ClusterNode

class CorrelationCluster:
    def __init__(self,num_features: int):
        """
        Args:
            num_features (int): number of features
        """
        self.num_features = num_features

        self.feature_sum = np.zeros(num_features)
        self.feature_residues_sum = np.zeros(num_features)
        self.feature_square_residues_sum = np.zeros(num_features)
        self.partial_correlations = np.zeros((num_features,num_features))
        self.update_count = 0 


    def update(self, x: np.ndarray):
        """
        Updates feature metrics based on one single entry (vector of features).

        Args:
            x (np.ndarray): Array of shape (num_features,)
        """
        self.update_count += 1
        self.feature_sum += x
        feature_residues_t = x - self.feature_sum/self.update_count
        self.feature_residues_sum += feature_residues_t
        self.feature_square_residues_sum += feature_residues_t**2
        self.partial_correlations += np.outer(feature_residues_t, feature_residues_t)


    def compute_correlation_distances(self):
        """
        Computes the correlation distance matrix between features.
        The correlation distance matrix is computed as

        D = 1 - C

        where C = sum(feat_i, feat_j) / sqrt(sum(feat_i**2)*sum(feat_j**2))
        is the Pearsson's correlation coefficient between features.
        """
        root_feature_square_residues_sum = np.sqrt(self.feature_square_residues_sum)
        denominator = np.outer(root_feature_square_residues_sum ,root_feature_square_residues_sum )
        denominator[denominator == 0] = 1e-100 #this protects against dive by zero erros (occurs when a feature is a constant)
        correlation_distances = 1-self.partial_correlations/denominator #the correlation distance matrix
        correlation_distances[correlation_distances < 0] = 0 #small negatives may appear due to the incremental fashion in which we update the mean. Therefore, we 'fix' them
        return correlation_distances


    def get_clusters(self, max_cluster_size: int):
        """
        Get clusters of <= max_cluster_size features. Clusters are formed based on the correlation distance matrix.

        Args:
            max_cluster_size (int): Maximum size of a feature cluster.
        """
        correlation_distances = self.compute_correlation_distances()
        linkages = linkage(
            correlation_distances[np.triu_indices(self.num_features, 1)]
        ) 
        if max_cluster_size < 1:
            max_cluster_size = 1
        if max_cluster_size > self.num_features:
            max_cluster_size = self.num_features
        cluster_tree = to_tree(linkages)
        clusters = self.break_clusters(cluster_tree,max_cluster_size)
        return clusters

    def break_clusters(self, cluster_tree: ClusterNode, max_cluster_size: int):
        """
        Recursively breaks clusters into subclusters until all subclusters have size <= max_cluster_size.

        Args:
            cluster_tree (ClusterNode): Root of the feature cluster tree.
            max_cluster_size (int): Maximum size of a feature cluster.
        """
        if cluster_tree.count <= max_cluster_size:  # base case: found a feature cluster of size <= max_cluster_size
            return [cluster_tree.pre_order()]  # return the ids of the features in the cluster
        return self.break_clusters(cluster_tree.get_left(),max_cluster_size) + self.break_clusters(cluster_tree.get_right(),max_cluster_size)