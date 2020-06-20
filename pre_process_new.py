import pandas as pd

def get_data(data, mode='train'):
    assert mode == 'train' or mode == 'test'

    if mode == 'train':
        data['vesselNextportETA'] = pd.to_datetime(data['vesselNextportETA'], infer_datetime_format=True,errors='coerce')
    elif mode == 'test':
        data['temp_timestamp'] = data['timestamp']
        data['onboardDate'] = pd.to_datetime(data['onboardDate'], infer_datetime_format=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], infer_datetime_format=True)
    data['longitude'] = data['longitude'].astype(float)
    data['loadingOrder'] = data['loadingOrder'].astype(str)
    data['latitude'] = data['latitude'].astype(float)
    data['speed'] = data['speed'].astype(float)
    data['direction'] = data['direction'].astype(float)

    return data


def get_feature(df, mode='train'):
    assert mode == 'train' or mode == 'test'

    df.sort_values(['loadingOrder', 'timestamp'], inplace=True)
    # 特征只选择经纬度、速度\方向
    df['lat_diff'] = df.groupby('loadingOrder')['latitude'].diff(1)
    df['lon_diff'] = df.groupby('loadingOrder')['longitude'].diff(1)
    df['speed_diff'] = df.groupby('loadingOrder')['speed'].diff(1)
    df['diff_minutes'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds() // 60
    df['anchor'] = df.apply(lambda x: 1 if x['lat_diff'] <= 0.03 and x['lon_diff'] <= 0.03
                                           and x['speed_diff'] <= 0.3 and x['diff_minutes'] <= 10 else 0, axis=1)



    if mode == 'train':
        group_df = df.groupby('loadingOrder')['timestamp'].agg(count='count',mmin = 'min')
        # 到“下一个港口”预计到达时间-最小时间戳，为label
        group_df['mmax'] = df.groupby('loadingOrder')['vesselNextportETA'].agg(mmax='max')['mmax']
        group_df['label'] = (group_df['mmax'] - group_df['mmin']).dt.total_seconds() / 3600.
        group_df = group_df.reset_index()
        group_df = group_df.drop(index=(group_df[group_df['count'] <= 1].index))
    elif mode == 'test':
        group_df = df.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count', mmin='min').reset_index()
        group_df = group_df.drop(index=(group_df[group_df['count'] <= 1].index))

    anchor_df = df.groupby('loadingOrder')['anchor'].agg('sum').reset_index()
    anchor_df.columns = ['loadingOrder', 'anchor_cnt']
    group_df = group_df.merge(anchor_df, on='loadingOrder', how='left')
    group_df['anchor_ratio'] = group_df['anchor_cnt'] / group_df['count']

    agg_function = ['min', 'max', 'mean', 'median']
    agg_col = ['latitude', 'longitude', 'speed', 'direction']

    group = df.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
    group.columns = ['loadingOrder'] + ['{}_{}'.format(i, j) for i in agg_col for j in agg_function]
    group_df = group_df.merge(group, on='loadingOrder', how='left')

    return group_df
