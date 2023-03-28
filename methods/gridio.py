import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm


def grid_io(df, blocks_num=10, io_interval=2):
    # clean data
    df.dropna(inplace=True)
    df = df[(df['speed'] >= 0) & (df['speed'] <= 150)]
    df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180) & (df['latitude'] >= -90) & (df['latitude'] <= 90)]
    df.drop_duplicates(subset=['id', 'datetime'], keep='first', inplace=True)
    df = df.sort_values(by=['id', 'datetime'])
    # drop speed error data
    speed_error_mask = (df['id'] == df['id'].shift(1)) & (df['longitude'] == df['longitude'].shift(1)) & \
                       (df['latitude'] == df['latitude'].shift(1)) & (df['speed'] != 0)
    df = df[~speed_error_mask]
    df.reset_index(drop=True, inplace=True)

    # get the boundary
    min_lng, max_lng = df['longitude'].min(), df['longitude'].max()
    min_lat, max_lat = df['latitude'].min(), df['latitude'].max()

    # separate blocks
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

    # separate time
    time_interval = datetime.timedelta(minutes=5)
    df = df.sort_values(by=['datetime'])
    time_groups = df.groupby(pd.Grouper(key='datetime', freq=time_interval))

    # calculate feature matrix
    feature_matrix = np.zeros(shape=(len(time_groups), len(block_ranges), 2))
    print('\nCalculating feature matrix')
    for i, (name, group) in tqdm(enumerate(time_groups)):
        group = group.reset_index(drop=True)
        for j, block in enumerate(block_ranges):

            block_mask = (group['longitude'] >= block['min_lng']) & (group['longitude'] < block['max_lng']) & \
                         (group['latitude'] >= block['min_lat']) & (group['latitude'] < block['max_lat'])
            group_block = group.loc[block_mask]
            num_taxis = len(group_block.groupby('id'))

            if num_taxis > 0:
                avg_speed = group_block['speed'].mean()
            else:
                avg_speed = 0
            feature_matrix[i, j, :] = [num_taxis, avg_speed]

    # save feature matrix
    print(feature_matrix.shape)
    np.save('./saved_files/feature_matrix_gi.npy', feature_matrix)
    print('feature matrix has been saved')

    # calculate in-out matrix
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

    # save in-out matrix
    print(io_matrix.shape)
    np.save('./saved_files/io_matrix_gi.npy', io_matrix)
    print('io matrix has been saved')

    # get adjacency matrix
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

    # save adjacency matrix
    adjacency_matrix = pd.DataFrame(adjacency_matrix)
    adjacency_matrix.to_csv('./saved_files/adjacency_matrix.csv', header=False)
    print('adjacency matrix has been saved')
