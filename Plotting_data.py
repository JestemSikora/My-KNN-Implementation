import matplotlib.pyplot as plt
import numpy as np

class PlottingData:
    def __init__(self, X, y, dataset, axis):
        self.X = X
        self.y = y
        self.dataset = dataset
        self.axis = axis

    def Plotting_Function(self, X, y, dataset, axis):
        repetition = 0
        X = X.reshape(1,len(X))
        for i in range(len(axis)):
            for j in range(len(axis)):
                repetition = repetition + 1
                idx_feature = 2 * i + j
                colors = np.random.rand(3,)

                axis[i,j].scatter(self.X[:,idx_feature], self.y, c=colors, s=8.5,
                                  label="Training Data")

                axis[i,j].scatter(X[:,idx_feature], y, c='red', s=30, label="Estimation")
                axis[i,j].set_xlabel(f"{dataset.feature_names[idx_feature]}")
                axis[i,j].set_ylabel("Class")
                axis[i,j].set_title(f"{repetition}th Feature - {dataset.feature_names[idx_feature]}")

                axis[i, j].legend(loc='upper right', fontsize=6)
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()

  
                   



