import numpy as np
import pandas as pd

file_path = './saved_files/'

adjacency_matrix_cluster = pd.read_csv(file_path + 'adjacency_matrix_cluster.csv', header=None)
print('adjacency_matrix_cluster:\n', adjacency_matrix_cluster.shape)

adjacency_matrix_grid= pd.read_csv(file_path + 'adjacency_matrix_grid.csv', header=None)
print('adjacency_matrix_grid:\n', adjacency_matrix_grid.shape)

dist_matrix_cluster= pd.read_csv(file_path + 'dist_matrix_cluster.csv', header=None)
print('dist_matrix_cluster:\n', dist_matrix_cluster.shape)

feature_matrix_cluster = np.load(file_path + 'feature_matrix_cluster.npy')
print('feature_matrix_cluster:\n', feature_matrix_cluster.shape)

feature_matrix_grid = np.load(file_path + 'feature_matrix_grid.npy')
print('feature_matrix_grid:\n', feature_matrix_grid.shape)

io_matrix_cluster = np.load(file_path + 'io_matrix_cluster.npy')
print('io_matrix_cluster:\n', io_matrix_cluster.shape)

io_matrix_grid = np.load(file_path + 'io_matrix_grid.npy')
print('io_matrix_grid:\n', io_matrix_grid.shape)