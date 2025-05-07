import numpy as np
from collections import Counter

class KNN_Algorithm:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def fit(self, X, y):
        first_X = X
        all_distances = self.Euclidean_distance(first_X, self.X)
        k_nearest = np.argsort(all_distances)[:self.k]
        voted = self.Vote(self.y, k_nearest)

        return voted

    def Euclidean_distance(self, first_X, X):
        all_distances = []

        for i in range(X.shape[0]):
            d = np.sqrt(np.sum((first_X - X[i]) ** 2))
            all_distances.append(d)

        return all_distances
    
    def Vote(self,y,k_nearest):
        nearest_labels = self.y[k_nearest]
        most_common = Counter(nearest_labels).most_common(1)[0][0]

        return most_common
    
    def prediction(self,X):
        first_X = X
        voted = self.fit(first_X, self.y)

        return voted



    

        

    
        

