"""
1.Use grid cells as nodes
2.Use the neighbours between one node to determine the value of the adjacency matrix(1/0)
"""
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm


def grid(df, blocks_num=10, fm_interval=5, io_interval=2):
    # Set parameters
    blocks_num = blocks_num
    fm_interval = fm_interval
    io_interval = io_interval
    # Get the boundary
    min_lng, max_lng = df['longitude'].min(), df['longitude'].max()
    min_lat, max_lat = df['latitude'].min(), df['latitude'].max()

    # Separate blocks
    n = blocks_num
    step_lng = (max_lng - min_lng) / n
    step_lat = (max_lat - min_lat) / n
    lng_list = np.arange(min_lng, max_lng + 1e-5, step_lng)
    lat_list = np.arange(min_lat, max_lat + 1e-5, step_lat)
    lat_list[-1] = lat_list[-1] + 1e-5
    lng_list[-1] = lng_list[-1] + 1e-5
    block_ranges = []
    for i in range(len(lat_list) - 1):
        for j in range(len(lng_list) - 1):
            block_ranges.append({'min_lng': lng_list[j], 'max_lng': lng_list[j + 1],
                                 'min_lat': lat_list[i], 'max_lat': lat_list[i + 1]})

    # Separate time
    time_interval = datetime.timedelta(minutes=fm_interval)
    df = df.sort_values(by=['datetime'])
    time_groups = df.groupby(pd.Grouper(key='datetime', freq=time_interval))

    # Calculate feature matrix
    feature_matrix = np.zeros(shape=(len(time_groups), len(block_ranges), 2))
    print('\nCalculating feature matrix')
    for i, (name, group) in tqdm(enumerate(time_groups)):
        group = group.reset_index(drop=True)
        for j, block in enumerate(block_ranges):

            block_mask = (group['longitude'] >= block['min_lng']) & (group['longitude'] < block['max_lng']) & \
                         (group['latitude'] >= block['min_lat']) & (group['latitude'] < block['max_lat'])
            group_block = group.loc[block_mask]
            # num_taxis = len(group_block.groupby('id'))
            num_taxis = len(group_block)

            if num_taxis > 0:
                avg_speed = group_block['speed'].mean()
            else:
                avg_speed = 0
            feature_matrix[i, j, :] = [num_taxis, avg_speed]

    # Save feature matrix
    print(feature_matrix.shape)
    np.save('./saved_files/feature_matrix_grid.npy', feature_matrix)
    print('feature matrix has been saved')

    # Calculate in-out matrix
    io_time_interval = datetime.timedelta(hours=io_interval)
    df_io = df.copy()
    df_io = df_io.sort_values(by=['datetime'])
    io_time_groups = df_io.groupby(pd.Grouper(key='datetime', freq=io_time_interval))

    io_matrix = np.zeros(shape=(len(io_time_groups), len(block_ranges), len(block_ranges), 2))
    print('\nCalculating in-out matrix')
    for t, (name, group) in tqdm(enumerate(io_time_groups)):
        for taxi_id, taxi_df in group.groupby('id'):

            taxi_df.reset_index(drop=True, inplace=True)
            lat_bins = np.digitize(taxi_df['latitude'], lat_list)
            lng_bins = np.digitize(taxi_df['longitude'], lng_list)
            block_trace = np.array((lat_bins - 1) * n + (lng_bins - 1))

            block_first_index = {f'{block_trace[0]}': 0}
            for i in range(len(block_trace) - 1):
                if block_trace[i] != block_trace[i + 1]:
                    block_first_index[f'{block_trace[i + 1]}'] = i + 1
                    start_block, end_block = block_trace[i], block_trace[i + 1]
                    io_matrix[t][start_block][end_block][0] += 1
                    io_matrix[t][start_block][end_block][1] += \
                        (taxi_df['datetime'].iloc[block_first_index[f'{block_trace[i + 1]}']] -
                         taxi_df['datetime'].iloc[block_first_index[f'{block_trace[i]}']]).seconds / 60

    io_matrix[:, :, :, 1] = io_matrix[:, :, :, 1] / (io_matrix[:, :, :, 0] + 1e-6)

    # Save in-out matrix
    print(io_matrix.shape)
    np.save('./saved_files/io_matrix_grid.npy', io_matrix)
    print('io matrix has been saved')

    # Get adjacency matrix
    adjacency_matrix = np.zeros((n ** 2, n ** 2))
    neighbors = []
    print('\nCalculating adjacency matrix')
    for i in tqdm(range(n)):
        for j in range(n):
            neighbor_ids = [(i + x, j + y) for x in [-1, 0, 1] for y in [-1, 0, 1] if
                            (0 <= i + x < n and 0 <= j + y < n)]
            neighbor_ids.remove((i, j))
            neighbor_ids = [idx[0] * n + idx[1] for idx in neighbor_ids]
            neighbors.append(neighbor_ids)
    dist = pd.DataFrame({"neighbors": neighbors})
    for i, row in dist.iterrows():
        adjacency_matrix[i, row["neighbors"]] = 1

    # Save adjacency matrix
    adjacency_matrix = pd.DataFrame(adjacency_matrix)
    adjacency_matrix.to_csv('./saved_files/adjacency_matrix_grid.csv', header=False, index=False)
    print('adjacency matrix has been saved')