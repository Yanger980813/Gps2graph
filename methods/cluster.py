"""
1.Use clustering to select main nodes
2.Use the distance between the main nodes to determine the value of the adjacency matrix(1/0)
"""
import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from einops import rearrange
from tqdm import tqdm


def cluster(df, nodes_num=50, fm_interval=5, io_interval=2, threshold=5.0):
    # Set parameters
    nodes_num = nodes_num
    fm_interval = fm_interval
    threshold = threshold
    # Clustering
    kmeans = KMeans(n_clusters=nodes_num)
    kmeans.fit(df[['longitude', 'latitude']])
    centroids = kmeans.cluster_centers_
    df['label'] = kmeans.predict(df[['longitude', 'latitude']])

    # Main nodes
    main_nodes = []
    for i in range(len(centroids)):
        main_nodes.append({'id': i, 'longitude': centroids[i][0], 'latitude': centroids[i][1]})

    # Distance
    print('\nCalculating distance matrix')
    dist_matrix = cdist(centroids, centroids, 'euclidean') * 111.32

    # Get adjacency matrix
    print('\nCalculating adjacency matrix')
    adjacency_matrix = np.zeros((len(centroids), len(centroids)), dtype=int)
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            if i == j:
                adjacency_matrix[i, j] = 0
            elif dist_matrix[i, j] < threshold:
                adjacency_matrix[i, j] = 1
            else:
                adjacency_matrix[i, j] = 0

    # Save adjacency matrix
    adjacency_matrix = pd.DataFrame(adjacency_matrix)
    adjacency_matrix.to_csv('./saved_files/adjacency_matrix_cluster.csv', header=False, index=False)
    # Save distance matrix
    dist_matrix = pd.DataFrame(dist_matrix)
    dist_matrix.to_csv('./saved_files/dist_matrix_cluster.csv', header=False, index=False)

    # Get feature matrix
    time_interval = datetime.timedelta(minutes=fm_interval)
    df = df.sort_values(by=['datetime'])
    groups = df.groupby(pd.Grouper(key='datetime', freq=time_interval))
    speeds = []
    flows = []

    # Nodes
    print('\nCalculating feature matrix')
    for i in tqdm(range(len(main_nodes))):
        node_speeds = []
        node_flows = []

        # Time and Labels
        for name, group in groups:
            group = group[group['label'] == i]
            # count = len(group.groupby('id'))
            count = len(group)
            if count > 0:
                speed = group['speed'].mean()
                node_speeds.append(speed)
                node_flows.append(count)
            else:
                node_speeds.append(0)
                node_flows.append(0)

        speeds.append(node_speeds)
        flows.append(node_flows)

    speeds = np.array(speeds)
    flows = np.array(flows)

    speeds = rearrange(speeds, 'n t -> t n 1')
    flows = rearrange(flows, 'n t -> t n 1')
    feature_matrix = np.concatenate([speeds, flows], axis=2)
    np.save('./saved_files/feature_matrix_cluster.npy', feature_matrix)

    # get io matrix
    io_time_interval = datetime.timedelta(hours=io_interval)
    df = df.sort_values(by=['datetime'])
    hour_groups = df.groupby(pd.Grouper(key='datetime', freq=io_time_interval))

    io_matrix = np.zeros((len(hour_groups), nodes_num, nodes_num, 2))

    print('\nCalculating in-out matrix')
    for hour, (name, hour_group) in tqdm(enumerate(hour_groups)):
        for taxi_id, taxi_df in hour_group.groupby('id'):
            nodes_list = np.array(taxi_df.label)
            nodes_first_index = {f'{nodes_list[0]}': 0}
            for i in range(len(nodes_list) - 1):
                if nodes_list[i] != nodes_list[i + 1]:
                    nodes_first_index[f'{nodes_list[i + 1]}'] = i + 1
                    start_label, end_label = nodes_list[i], nodes_list[i + 1]
                    io_matrix[hour][start_label][end_label][0] += 1
                    io_matrix[hour][start_label][end_label][1] += \
                        (taxi_df['datetime'].iloc[nodes_first_index[f'{nodes_list[i + 1]}']] -
                         taxi_df['datetime'].iloc[nodes_first_index[f'{nodes_list[i]}']]).seconds / 60

    io_matrix[:, :, :, 1] = io_matrix[:, :, :, 1] / (io_matrix[:, :, :, 0] + 1e-5)
    np.save('./saved_files/io_matrix_cluster.npy', io_matrix)