import numpy as np

# Calculate speed(if necessary)
def cal_speed(df):
    time_diff = df['datetime'].diff().dt.total_seconds().fillna(0)
    distance_diff = np.sqrt(((df['latitude'].diff() * 111320) ** 2) + ((df['longitude'].diff() * 111320) ** 2)).fillna \
        (0)
    df['speed'] = distance_diff / time_diff
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df