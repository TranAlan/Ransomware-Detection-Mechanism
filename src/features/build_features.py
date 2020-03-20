#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
    This is the main file to making and engineering features.
'''

from os import makedirs
from os.path import dirname

import math
import sys
import time
import pandas as pd
import numpy as np

sys.path.append('../')
from utils.file_util import load_yaml

#Global
CONFIG_PATH = './features_config.yml'

def main():
    """
        Retrieves an already preprocessed version of the dataset.
        The final CSV will be sorted on StartTime.
        Build DstBytes
        Build features based on the following:
            Total flows in the forward direction in the window
            Total flows in the backward direction in the window
            Total size of netflows in forward direction in the window
            Total size of netflows in backward direction in the window
            Minimum size of flow in forward direction in the window
            Minimum size of flow in backward direction in the window
            Maximum size of flow in forward direction in the window
            Maximum size of flow in backward direction in the window
            Mean size of flow in forward direction in the window
            Mean size of flow in backward direction in the window
            Standard Deviation size of flow in forward direction in the window
            Standard Deviation size of flow in backward direction in the window
            Time between 2 flows in the window in the forward direction
            Time between 2 flows in the window in the backward direction
        A similar approach is down on TotBytes, TotPkts, SrcBytes.
        Window is 10k elements and 10 Minutes.
        This window is done again with focus on source and destination addresses
        A sample containing the first 50 rows will be saved.
        A new CSV with raw + discretized + engineered will be saved.
    """
    start = time.time()
    config = load_yaml(CONFIG_PATH)
    preprocessed_path = config['preprocessed_path']
    processed_output_path = config['processed_path']
    sample_output_path = config['sample_processed_path']
    sample_size = config['sample_size']
    preprocessed_df = pd.read_csv(preprocessed_path)
    feature_df = preprocessed_df.copy()
    feature_df['StartTime'] = pd.to_datetime(feature_df['StartTime'])
    feature_df = feature_df.sort_values('StartTime', ignore_index=True)
    feature_df['epoch'] = ((feature_df['StartTime'] - pd.Timestamp("1970-01-01"))
                           // pd.Timedelta('1ms')) / 1000
    #Window for N elements
    window_n_elements = 10000
    window_time_min = 10
    window_time_rolling_window = '10T'
    #Bytes
    feature_df['DstBytes'] = feature_df['TotBytes'] - feature_df['SrcBytes']

    print(f'Building Total Flow in {window_time_min} minutes')
    feature_df = build_tot_flows_time_window(feature_df, '10T')
    print(f'Building TotBytes, TotPkts, SrcBytes, metrics in {window_time_min} minutes')
    feature_df = build_time_features(feature_df, 'TotBytes', 'TotB', window_time_rolling_window)
    feature_df = build_time_features(feature_df, 'TotPkts', 'TotPkt', window_time_rolling_window)
    feature_df = build_time_features(feature_df, 'SrcBytes', 'SrcB', window_time_rolling_window)

    print(f'Building TotBytes, TotPkts, SrcBytes with {window_n_elements} elements on SrcAddr')
    feature_df = build_pkts_bytes_x_window(feature_df, window_n_elements, 'SrcAddr')
    print()
    print(f'Building TotBytes, TotPkts, SrcBytes with {window_n_elements} elements on DstAddr')
    feature_df = build_pkts_bytes_x_window(feature_df, window_n_elements, 'DstAddr')
    print()
    print(f'Building TotBytes, TotPkts, SrcBytes with {window_time_min} minutes on SrcAddr')
    feature_df = build_pkts_bytes_time_window(feature_df, window_time_min, 'SrcAddr')
    print()
    print(f'Building TotBytes, TotPkts, SrcBytes with {window_time_min} minutes on DstAddr')
    feature_df = build_pkts_bytes_time_window(feature_df, window_time_min, 'DstAddr')
    print()
    print(f'Building Time Between 2 Flows with {window_time_min} minutes')
    feature_df = build_time_bet_2_flow_time_window(feature_df, window_time_min)
    print()

    #Write Sample to CSV
    makedirs(dirname(processed_output_path), exist_ok=True)
    feature_df = feature_df.drop(columns=['epoch']) #Remove as it is not part of the feature list
    feature_df.head(sample_size).to_csv(sample_output_path, index=False)
    #Write Raw and Features to CSV file.
    feature_df.to_csv(processed_output_path, index=False)
    print(f'Time Elapsed: {time.time() - start}')

def addr_prefix_name(addr):
    '''
        Returns (String)
        Returns S if addr is SrcAddr
        Returns D if addr is DstAddr
    '''
    if addr == 'SrcAddr':
        return 'S'
    if addr == 'DstAddr':
        return 'D'
    return ''

def convert_to_int_64(data_f, columns):
    '''
        Given a dataframe and a list of strings of columns in dataframe,
        try to convert each column datatype to int64.
    '''
    copy = data_f.copy()
    for col in columns:
        copy[col] = copy[col].astype('int64')
    return copy

def create_fwd_bwd_col(f_df, col, name):
    '''
        Given a column, it will create 2 extra columns, fwd, bwd
        of that column, to allow rolling on it.
        Ex. The column will fwd name will have all 0s in rows where
        is_fwd is 0 instead of its original value.
    '''
    data_f = f_df.copy()

    #Forward direction
    fwd_name = f'{name}_fwd'
    data_f[fwd_name] = data_f[col]
    data_f.loc[data_f.is_fwd == 0, fwd_name] = np.NaN

    #Backword direction
    bwd_name = f'{name}_bwd'
    data_f[bwd_name] = data_f[col]
    data_f.loc[data_f.is_fwd == 1, bwd_name] = np.NaN
    return data_f

def helper_calc_flow_diff(epoch_list, window_range_sec):
    '''
        If there is 2 items, calculate the time difference.
        Return difference if it is less than window range sec.
    '''
    if len(epoch_list) < 2 or (epoch_list[-1] - epoch_list[-2]) > window_range_sec:
        return 0
    return epoch_list[-1] - epoch_list[-2]

def helper_trim_dict_row_list(key, dictionary, curr_epoch, time_window):
    '''
        Assumes the value of the dictionary is a list.
        While given date - the oldest date is larger then time window,
        pop list.
    '''
    if key in dictionary:
        while len(dictionary[key]) > 0 and curr_epoch - dictionary[key][-1].epoch > time_window:
            dictionary[key].pop()
    else:
        dictionary[key] = []

def helper_trim_dict_value_list(key, dictionary, size):
    '''
        Assumes the value of the dictionary is a list.
        If list exceeds size. Pop until list conforms with size.
        If key does not exist. Create a key value pair with empty list.
    '''
    if key in dictionary:
        while len(dictionary[key]) >= size:
            dictionary[key].pop(0)
    else:
        dictionary[key] = []

def print_status(index, total, percentage=5):
    '''
        Prints the number of iterations before
        a loop is completed given an index, total len
        of loop, and percentage.
    '''
    threshold = math.ceil(total*(percentage/100))
    if index % threshold == 0:
        completion = index/total * 100
        print(f'TASK: {completion}% Completed')

def build_tot_flows_time_window(f_df, roll_min):
    '''
        Total Flow in forward and backward 10 min.
        Must give pandas rolling time.
        10min = 10T
    '''
    data_f = f_df.copy()
    df_with_bwd = data_f.copy()
    df_with_bwd['is_bwd'] = df_with_bwd['is_fwd'].replace({0:1, 1:0})
    fwd_rolling = data_f[['StartTime', 'is_fwd']].rolling(roll_min, on='StartTime')
    data_f[f'TotFlowFwd_{roll_min}'] = fwd_rolling.sum()['is_fwd']
    bwd_rolling = df_with_bwd[['StartTime', 'is_bwd']].rolling(roll_min, on='StartTime')
    data_f[f'TotFlowBwd_{roll_min}'] = bwd_rolling.sum()['is_bwd']
    return convert_to_int_64(data_f, [f'TotFlowFwd_{roll_min}', f'TotFlowBwd_{roll_min}'])

def build_time_features(f_df, col_name, short_name, roll_min): # pylint: disable=R0914
    '''
        Time should be pandas rolling period.
        Builds a given column the min, max, mean, std given a pandas rolling time.
    '''
    data_f = f_df.copy()

    #Require a df with exta attributes b/c of the rolling.
    #See create_fwd_bwd_col
    #Forward direction and backward of the given column.
    df_rolling = create_fwd_bwd_col(data_f, col_name, short_name)

    fwd_col = short_name + '_fwd'
    bwd_col = short_name + '_bwd'

    time_window = df_rolling[['StartTime', fwd_col, bwd_col]].rolling(roll_min, on='StartTime')

    #Total Size in Forward and Backward based on time
    sums = time_window.sum()
    sum_fwd_name, sum_bwd_name = f'{short_name}SumFwd_{roll_min}', f'{short_name}SumBwd_{roll_min}'
    data_f[sum_fwd_name] = sums[fwd_col]
    data_f[sum_bwd_name] = sums[bwd_col]

    #Min Size in Forward and Backward based on time
    mins = time_window.min()
    min_fwd_name, min_bwd_name = f'{short_name}MinFwd_{roll_min}', f'{short_name}MinBwd_{roll_min}'
    data_f[min_fwd_name] = mins[fwd_col]
    data_f[min_bwd_name] = mins[bwd_col]

    #Max Size in Forward and Backward based on time
    maxs = time_window.max()
    max_fwd_name, max_bwd_name = f'{short_name}MaxFwd_{roll_min}', f'{short_name}MaxBwd_{roll_min}'
    data_f[max_fwd_name] = maxs[fwd_col]
    data_f[max_bwd_name] = maxs[bwd_col]

    #Mean Size in Forward and Backward based on time
    means = time_window.mean()
    mean_fwd_name = f'{short_name}MeanFwd_{roll_min}'
    mean_bwd_name = f'{short_name}MeanBwd_{roll_min}'
    data_f[mean_fwd_name] = means[fwd_col]
    data_f[mean_bwd_name] = means[bwd_col]

    #Standard Deviation Size in Forward and Backward based on time
    stds = time_window.std()
    std_fwd_name, std_bwd_name = f'{short_name}StdFwd_{roll_min}', f'{short_name}StdBwd_{roll_min}'
    data_f[std_fwd_name] = stds[fwd_col]
    data_f[std_bwd_name] = stds[bwd_col]
    #Fill all columns with np.NaN with zero
    data_f[[sum_fwd_name,
            sum_bwd_name,
            min_fwd_name,
            min_bwd_name,
            max_fwd_name,
            max_bwd_name,
            mean_fwd_name,
            mean_bwd_name,
            std_fwd_name,
            std_bwd_name]] = data_f[[sum_fwd_name,
                                     sum_bwd_name,
                                     min_fwd_name,
                                     min_bwd_name,
                                     max_fwd_name,
                                     max_bwd_name,
                                     mean_fwd_name,
                                     mean_bwd_name,
                                     std_fwd_name,
                                     std_bwd_name]].fillna(0)

    #All columns that can be int, convert to int.
    return convert_to_int_64(data_f,
                             [sum_fwd_name,
                              sum_bwd_name,
                              min_fwd_name,
                              min_bwd_name,
                              max_fwd_name,
                              max_bwd_name])

def build_pkts_bytes_time_window(f_df, minute, addr): # pylint: disable=R0914, R0915
    '''
        Creates new columns for the sum, min, max, mean, std()
        of TotBytes, TotPkts, SrcBytes within a given time window,
        and which if specific to SrcAddr or DstAddr
    '''
    data_f = f_df[[addr, 'is_fwd', 'TotPkts', 'TotBytes', 'SrcBytes', 'epoch']].copy()
    total = len(data_f.index)
    window_range_sec = minute * 60
    prefix = addr_prefix_name(addr)

    #Total Flow
    tot_flow_fwd_l = []
    tot_flow_bwd_l = []

    #Total Bytes Forward
    tot_b_sum_fwd_l = []
    tot_b_min_fwd_l = []
    tot_b_max_fwd_l = []
    tot_b_mean_fwd_l = []
    tot_b_std_fwd_l = []

    #Total Bytes Backward
    tot_b_sum_bwd_l = []
    tot_b_min_bwd_l = []
    tot_b_max_bwd_l = []
    tot_b_mean_bwd_l = []
    tot_b_std_bwd_l = []

    #Total Packets Forward
    tot_pkt_sum_fwd_l = []
    tot_pkt_min_fwd_l = []
    tot_pkt_max_fwd_l = []
    tot_pkt_mean_fwd_l = []
    tot_pkt_std_fwd_l = []

    #Total Packets Backward
    tot_pkt_sum_bwd_l = []
    tot_pkt_min_bwd_l = []
    tot_pkt_max_bwd_l = []
    tot_pkt_mean_bwd_l = []
    tot_pkt_std_bwd_l = []

    #Total Src Bytes Forward
    src_b_sum_fwd_l = []
    src_b_min_fwd_l = []
    src_b_max_fwd_l = []
    src_b_mean_fwd_l = []
    src_b_std_fwd_l = []

    #Total Src Bytes Backward
    src_b_sum_bwd_l = []
    src_b_min_bwd_l = []
    src_b_max_bwd_l = []
    src_b_mean_bwd_l = []
    src_b_std_bwd_l = []

    fwd_dict = {}
    bwd_dict = {}

    for index, row in data_f.iterrows():
        print_status(index, total)

        #Independent whether the row itself is fwd or backward
        helper_trim_dict_row_list(row[addr], fwd_dict, row.epoch, window_range_sec)
        helper_trim_dict_row_list(row[addr], bwd_dict, row.epoch, window_range_sec)
        fwd_rows_list = fwd_dict[row[addr]]
        bwd_rows_list = bwd_dict[row[addr]]

        if row.is_fwd:
            fwd_rows_list.insert(0, row)
        else:
            bwd_rows_list.insert(0, row)

        if len(fwd_rows_list) == 0:
            fwd_df = pd.DataFrame(data={
                'TotPkts': [],
                'TotBytes': [],
                'SrcBytes': []
            })
        else:
            fwd_df = pd.DataFrame(data=fwd_rows_list)

        if len(bwd_rows_list) == 0:
            bwd_df = pd.DataFrame(data={
                'TotPkts': [],
                'TotBytes': [],
                'SrcBytes': []
            })
        else:
            bwd_df = pd.DataFrame(data=bwd_rows_list)

        #Total Flow
        tot_flow_fwd_l.append(len(fwd_rows_list))
        tot_flow_bwd_l.append(len(bwd_rows_list))

        if len(fwd_rows_list) != 0:
            #Total Bytes Fwd
            tot_b_fwd_l = fwd_df.TotBytes
            tot_b_sum_fwd_l.append(tot_b_fwd_l.sum())
            tot_b_min_fwd_l.append(tot_b_fwd_l.min())
            tot_b_max_fwd_l.append(tot_b_fwd_l.max())
            tot_b_mean_fwd_l.append(tot_b_fwd_l.mean())
            tot_b_std_fwd_l.append(tot_b_fwd_l.std() if len(tot_b_fwd_l) > 1 else 0)

            #Total Packets Fwd
            tot_pkts_fwd_l = fwd_df.TotPkts
            tot_pkt_sum_fwd_l.append(tot_pkts_fwd_l.sum())
            tot_pkt_min_fwd_l.append(tot_pkts_fwd_l.min())
            tot_pkt_max_fwd_l.append(tot_pkts_fwd_l.max())
            tot_pkt_mean_fwd_l.append(tot_pkts_fwd_l.mean())
            tot_pkt_std_fwd_l.append(tot_pkts_fwd_l.std() if len(tot_pkts_fwd_l) > 1 else 0)

            #Total Source Bytes Fwd
            src_b_fwd_l = fwd_df.SrcBytes
            src_b_sum_fwd_l.append(src_b_fwd_l.sum())
            src_b_min_fwd_l.append(src_b_fwd_l.min())
            src_b_max_fwd_l.append(src_b_fwd_l.max())
            src_b_mean_fwd_l.append(src_b_fwd_l.mean())
            src_b_std_fwd_l.append(src_b_fwd_l.std() if len(src_b_fwd_l) > 1 else 0)
        else:
            #Total Bytes Fwd
            tot_b_fwd_l = fwd_df.TotBytes
            tot_b_sum_fwd_l.append(0)
            tot_b_min_fwd_l.append(0)
            tot_b_max_fwd_l.append(0)
            tot_b_mean_fwd_l.append(0)
            tot_b_std_fwd_l.append(tot_b_fwd_l.std() if len(tot_b_fwd_l) > 1 else 0)

            #Total Packets Fwd
            tot_pkts_fwd_l = fwd_df.TotPkts
            tot_pkt_sum_fwd_l.append(0)
            tot_pkt_min_fwd_l.append(0)
            tot_pkt_max_fwd_l.append(0)
            tot_pkt_mean_fwd_l.append(0)
            tot_pkt_std_fwd_l.append(0)

            #Total Source Bytes Fwd
            src_b_fwd_l = fwd_df.SrcBytes
            src_b_sum_fwd_l.append(0)
            src_b_min_fwd_l.append(0)
            src_b_max_fwd_l.append(0)
            src_b_mean_fwd_l.append(0)
            src_b_std_fwd_l.append(0)

        if len(bwd_rows_list) != 0:
            #Total Bytes Bwd
            tot_b_bwd_l = bwd_df.TotBytes
            tot_b_sum_bwd_l.append(tot_b_bwd_l.sum())
            tot_b_min_bwd_l.append(tot_b_bwd_l.min())
            tot_b_max_bwd_l.append(tot_b_bwd_l.max())
            tot_b_mean_bwd_l.append(tot_b_bwd_l.mean())
            tot_b_std_bwd_l.append(tot_b_bwd_l.std() if len(tot_b_bwd_l) > 1 else 0)

            #Total Packets Bwd
            tot_pkts_bwd_l = bwd_df.TotPkts
            tot_pkt_sum_bwd_l.append(tot_pkts_bwd_l.sum())
            tot_pkt_min_bwd_l.append(tot_pkts_bwd_l.min())
            tot_pkt_max_bwd_l.append(tot_pkts_bwd_l.max())
            tot_pkt_mean_bwd_l.append(tot_pkts_bwd_l.mean())
            tot_pkt_std_bwd_l.append(tot_pkts_bwd_l.std() if len(tot_pkts_bwd_l) > 1 else 0)

            #Total Source Bytes Bwd
            src_b_bwd_l = bwd_df.SrcBytes
            src_b_sum_bwd_l.append(src_b_bwd_l.sum())
            src_b_min_bwd_l.append(src_b_bwd_l.min())
            src_b_max_bwd_l.append(src_b_bwd_l.max())
            src_b_mean_bwd_l.append(src_b_bwd_l.mean())
            src_b_std_bwd_l.append(src_b_bwd_l.std() if len(src_b_bwd_l) > 1 else 0)
        else:
            #Total Bytes Bwd
            tot_b_bwd_l = bwd_df.TotBytes
            tot_b_sum_bwd_l.append(0)
            tot_b_min_bwd_l.append(0)
            tot_b_max_bwd_l.append(0)
            tot_b_mean_bwd_l.append(0)
            tot_b_std_bwd_l.append(0)

            #Total Packets Bwd
            tot_pkts_bwd_l = bwd_df.TotPkts
            tot_pkt_sum_bwd_l.append(0)
            tot_pkt_min_bwd_l.append(0)
            tot_pkt_max_bwd_l.append(0)
            tot_pkt_mean_bwd_l.append(0)
            tot_pkt_std_bwd_l.append(0)


            #Total Source Bytes Bwd
            src_b_bwd_l = bwd_df.SrcBytes
            src_b_sum_bwd_l.append(0)
            src_b_min_bwd_l.append(0)
            src_b_max_bwd_l.append(0)
            src_b_mean_bwd_l.append(0)
            src_b_std_bwd_l.append(0)

    #Build Data Frame
    new_features_df = pd.DataFrame(data={
        f'{prefix}TotFlowFwd_{minute}T': tot_flow_fwd_l,
        f'{prefix}TotFlowBwd_{minute}T': tot_flow_bwd_l,
        f'{prefix}TotBSumFwd_{minute}T': tot_b_sum_fwd_l,
        f'{prefix}TotBMinFwd_{minute}T': tot_b_min_fwd_l,
        f'{prefix}TotBMaxFwd_{minute}T': tot_b_max_fwd_l,
        f'{prefix}TotBMeanFwd_{minute}T': tot_b_mean_fwd_l,
        f'{prefix}TotBStdFwd_{minute}T': tot_b_std_fwd_l,
        f'{prefix}TotBSumBwd_{minute}T': tot_b_sum_bwd_l,
        f'{prefix}TotBMinBwd_{minute}T': tot_b_min_bwd_l,
        f'{prefix}TotBMaxBwd_{minute}T': tot_b_max_bwd_l,
        f'{prefix}TotBMeanBwd_{minute}T': tot_b_mean_bwd_l,
        f'{prefix}TotBStdBwd_{minute}T': tot_b_std_bwd_l,
        f'{prefix}TotPktSumFwd_{minute}T': tot_pkt_sum_fwd_l,
        f'{prefix}TotPktMinFwd_{minute}T': tot_pkt_min_fwd_l,
        f'{prefix}TotPktMaxFwd_{minute}T': tot_pkt_max_fwd_l,
        f'{prefix}TotPktMeanFwd_{minute}T': tot_pkt_mean_fwd_l,
        f'{prefix}TotPktStdFwd_{minute}T': tot_pkt_std_fwd_l,
        f'{prefix}TotPktSumBwd_{minute}T': tot_pkt_sum_bwd_l,
        f'{prefix}TotPktMinBwd_{minute}T': tot_pkt_min_bwd_l,
        f'{prefix}TotPktMaxBwd_{minute}T': tot_pkt_max_bwd_l,
        f'{prefix}TotPktMeanBwd_{minute}T': tot_pkt_mean_bwd_l,
        f'{prefix}TotPktStdBwd_{minute}T': tot_pkt_std_bwd_l,
        f'{prefix}SrcBSumFwd_{minute}T': src_b_sum_fwd_l,
        f'{prefix}SrcBMinFwd_{minute}T': src_b_min_fwd_l,
        f'{prefix}SrcBMaxFwd_{minute}T': src_b_max_fwd_l,
        f'{prefix}SrcBMeanFwd_{minute}T': src_b_mean_fwd_l,
        f'{prefix}SrcBStdFwd_{minute}T': src_b_std_fwd_l,
        f'{prefix}SrcBSumBwd_{minute}T': src_b_sum_bwd_l,
        f'{prefix}SrcBMinBwd_{minute}T': src_b_min_bwd_l,
        f'{prefix}SrcBMaxBwd_{minute}T': src_b_max_bwd_l,
        f'{prefix}SrcBMeanBwd_{minute}T': src_b_mean_bwd_l,
        f'{prefix}SrcBStdBwd_{minute}T': src_b_std_bwd_l
    })
    return pd.concat([f_df, new_features_df], axis=1)

def build_pkts_bytes_x_window(f_df, num, addr): # pylint: disable=R0914, R0915
    '''
        Creates new columns for the sum, min, max, mean, std()
        of TotBytes, TotPkts, SrcBytes within window of num elements,
        and which if specific to SrcAddr or DstAddr
    '''
    data_f = f_df[[addr, 'is_fwd', 'TotPkts', 'TotBytes', 'SrcBytes', 'epoch']].copy()
    total_length = len(data_f.index)
    window = []
    prefix = addr_prefix_name(addr)

    #Total Flow
    tot_flow_fwd_l = []
    tot_flow_bwd_l = []

    #Total Bytes Forward
    tot_b_sum_fwd_l = []
    tot_b_min_fwd_l = []
    tot_b_max_fwd_l = []
    tot_b_mean_fwd_l = []
    tot_b_std_fwd_l = []

    #Total Bytes Backward
    tot_b_sum_bwd_l = []
    tot_b_min_bwd_l = []
    tot_b_max_bwd_l = []
    tot_b_mean_bwd_l = []
    tot_b_std_bwd_l = []

    #Total Packets Forward
    tot_pkt_sum_fwd_l = []
    tot_pkt_min_fwd_l = []
    tot_pkt_max_fwd_l = []
    tot_pkt_mean_fwd_l = []
    tot_pkt_std_fwd_l = []

    #Total Packets Backward
    tot_pkt_sum_bwd_l = []
    tot_pkt_min_bwd_l = []
    tot_pkt_max_bwd_l = []
    tot_pkt_mean_bwd_l = []
    tot_pkt_std_bwd_l = []

    #Total Src Bytes Forward
    src_b_sum_fwd_l = []
    src_b_min_fwd_l = []
    src_b_max_fwd_l = []
    src_b_mean_fwd_l = []
    src_b_std_fwd_l = []

    #Total Src Bytes Backward
    src_b_sum_bwd_l = []
    src_b_min_bwd_l = []
    src_b_max_bwd_l = []
    src_b_mean_bwd_l = []
    src_b_std_bwd_l = []

    #Time Between Flows
    time_flow_fwd_sec_l = []
    time_flow_bwd_sec_l = []

    for index, row in data_f.iterrows():
        print_status(index, total_length)

        #Add row to window
        if len(window) == num:
            window.pop(0)
        window.append(row)

        #Get all rows with the same address and in forward or backward directions.
        current_row_addr = row[addr]

        window_df = pd.DataFrame(data=window)
        addr_df = window_df.loc[window_df[addr] == current_row_addr]

        fwd_df = addr_df.loc[addr_df.is_fwd == 1]
        bwd_df = addr_df.loc[addr_df.is_fwd == 0]

        #Total Flow
        tot_flow_fwd_l.append(len(fwd_df))
        tot_flow_bwd_l.append(len(bwd_df))

        if len(fwd_df) != 0:
            #Total Bytes Fwd
            tot_b_fwd_l = fwd_df.TotBytes
            tot_b_sum_fwd_l.append(tot_b_fwd_l.sum())
            tot_b_min_fwd_l.append(tot_b_fwd_l.min())
            tot_b_max_fwd_l.append(tot_b_fwd_l.max())
            tot_b_mean_fwd_l.append(tot_b_fwd_l.mean())
            tot_b_std_fwd_l.append(tot_b_fwd_l.std() if len(tot_b_fwd_l) > 1 else 0)

            #Total Packets Fwd
            tot_pkts_fwd_l = fwd_df.TotPkts
            tot_pkt_sum_fwd_l.append(tot_pkts_fwd_l.sum())
            tot_pkt_min_fwd_l.append(tot_pkts_fwd_l.min())
            tot_pkt_max_fwd_l.append(tot_pkts_fwd_l.max())
            tot_pkt_mean_fwd_l.append(tot_pkts_fwd_l.mean())
            tot_pkt_std_fwd_l.append(tot_pkts_fwd_l.std() if len(tot_pkts_fwd_l) > 1 else 0)

            #Total Source Bytes Fwd
            src_b_fwd_l = fwd_df.SrcBytes
            src_b_sum_fwd_l.append(src_b_fwd_l.sum())
            src_b_min_fwd_l.append(src_b_fwd_l.min())
            src_b_max_fwd_l.append(src_b_fwd_l.max())
            src_b_mean_fwd_l.append(src_b_fwd_l.mean())
            src_b_std_fwd_l.append(src_b_fwd_l.std() if len(src_b_fwd_l) > 1 else 0)
        else:
            #Total Bytes Fwd
            tot_b_fwd_l = fwd_df.TotBytes
            tot_b_sum_fwd_l.append(0)
            tot_b_min_fwd_l.append(0)
            tot_b_max_fwd_l.append(0)
            tot_b_mean_fwd_l.append(0)
            tot_b_std_fwd_l.append(tot_b_fwd_l.std() if len(tot_b_fwd_l) > 1 else 0)

            #Total Packets Fwd
            tot_pkts_fwd_l = fwd_df.TotPkts
            tot_pkt_sum_fwd_l.append(0)
            tot_pkt_min_fwd_l.append(0)
            tot_pkt_max_fwd_l.append(0)
            tot_pkt_mean_fwd_l.append(0)
            tot_pkt_std_fwd_l.append(0)

            #Total Source Bytes Fwd
            src_b_fwd_l = fwd_df.SrcBytes
            src_b_sum_fwd_l.append(0)
            src_b_min_fwd_l.append(0)
            src_b_max_fwd_l.append(0)
            src_b_mean_fwd_l.append(0)
            src_b_std_fwd_l.append(0)

        if len(bwd_df) != 0:
            #Total Bytes Bwd
            tot_b_bwd_l = bwd_df.TotBytes
            tot_b_sum_bwd_l.append(tot_b_bwd_l.sum())
            tot_b_min_bwd_l.append(tot_b_bwd_l.min())
            tot_b_max_bwd_l.append(tot_b_bwd_l.max())
            tot_b_mean_bwd_l.append(tot_b_bwd_l.mean())
            tot_b_std_bwd_l.append(tot_b_bwd_l.std() if len(tot_b_bwd_l) > 1 else 0)

            #Total Packets Bwd
            tot_pkts_bwd_l = bwd_df.TotPkts
            tot_pkt_sum_bwd_l.append(tot_pkts_bwd_l.sum())
            tot_pkt_min_bwd_l.append(tot_pkts_bwd_l.min())
            tot_pkt_max_bwd_l.append(tot_pkts_bwd_l.max())
            tot_pkt_mean_bwd_l.append(tot_pkts_bwd_l.mean())
            tot_pkt_std_bwd_l.append(tot_pkts_bwd_l.std() if len(tot_pkts_bwd_l) > 1 else 0)

            #Total Source Bytes Bwd
            src_b_bwd_l = bwd_df.SrcBytes
            src_b_sum_bwd_l.append(src_b_bwd_l.sum())
            src_b_min_bwd_l.append(src_b_bwd_l.min())
            src_b_max_bwd_l.append(src_b_bwd_l.max())
            src_b_mean_bwd_l.append(src_b_bwd_l.mean())
            src_b_std_bwd_l.append(src_b_bwd_l.std() if len(src_b_bwd_l) > 1 else 0)
        else:
            #Total Bytes Bwd
            tot_b_bwd_l = bwd_df.TotBytes
            tot_b_sum_bwd_l.append(0)
            tot_b_min_bwd_l.append(0)
            tot_b_max_bwd_l.append(0)
            tot_b_mean_bwd_l.append(0)
            tot_b_std_bwd_l.append(0)

            #Total Packets Bwd
            tot_pkts_bwd_l = bwd_df.TotPkts
            tot_pkt_sum_bwd_l.append(0)
            tot_pkt_min_bwd_l.append(0)
            tot_pkt_max_bwd_l.append(0)
            tot_pkt_mean_bwd_l.append(0)
            tot_pkt_std_bwd_l.append(0)

            #Total Source Bytes Bwd
            src_b_bwd_l = bwd_df.SrcBytes
            src_b_sum_bwd_l.append(0)
            src_b_min_bwd_l.append(0)
            src_b_max_bwd_l.append(0)
            src_b_mean_bwd_l.append(0)
            src_b_std_bwd_l.append(0)

        #Time Between Flows
        if len(fwd_df) < 2:
            time_flow_fwd_sec_l.append(0)
        else:
            time_flow_fwd_sec_l.append(fwd_df.iloc[-1].epoch - fwd_df.iloc[-2].epoch)

        if len(bwd_df) < 2:
            time_flow_bwd_sec_l.append(0)
        else:
            time_flow_bwd_sec_l.append(bwd_df.iloc[-1].epoch - bwd_df.iloc[-2].epoch)

    #Build Data Frame
    new_features_df = pd.DataFrame(data={
        f'{prefix}TotFlowFwdN_{num}': tot_flow_fwd_l,
        f'{prefix}TotFlowBwdN_{num}': tot_flow_bwd_l,
        f'{prefix}TotBSumFwdN_{num}': tot_b_sum_fwd_l,
        f'{prefix}TotBMinFwdN_{num}': tot_b_min_fwd_l,
        f'{prefix}TotBMaxFwdN_{num}': tot_b_max_fwd_l,
        f'{prefix}TotBMeanFwdN_{num}': tot_b_mean_fwd_l,
        f'{prefix}TotBStdFwdN_{num}': tot_b_std_fwd_l,
        f'{prefix}TotBSumBwdN_{num}': tot_b_sum_bwd_l,
        f'{prefix}TotBMinBwdN_{num}': tot_b_min_bwd_l,
        f'{prefix}TotBMaxBwdN_{num}': tot_b_max_bwd_l,
        f'{prefix}TotBMeanBwdN_{num}': tot_b_mean_bwd_l,
        f'{prefix}TotBStdBwdN_{num}': tot_b_std_bwd_l,
        f'{prefix}TotPktSumFwdN_{num}': tot_pkt_sum_fwd_l,
        f'{prefix}TotPktMinFwdN_{num}': tot_pkt_min_fwd_l,
        f'{prefix}TotPktMaxFwdN_{num}': tot_pkt_max_fwd_l,
        f'{prefix}TotPktMeanFwdN_{num}': tot_pkt_mean_fwd_l,
        f'{prefix}TotPktStdFwdN_{num}': tot_pkt_std_fwd_l,
        f'{prefix}TotPktSumBwdN_{num}': tot_pkt_sum_bwd_l,
        f'{prefix}TotPktMinBwdN_{num}': tot_pkt_min_bwd_l,
        f'{prefix}TotPktMaxBwdN_{num}': tot_pkt_max_bwd_l,
        f'{prefix}TotPktMeanBwdN_{num}': tot_pkt_mean_bwd_l,
        f'{prefix}TotPktStdBwdN_{num}': tot_pkt_std_bwd_l,
        f'{prefix}SrcBSumFwdN_{num}': src_b_sum_fwd_l,
        f'{prefix}SrcBMinFwdN_{num}': src_b_min_fwd_l,
        f'{prefix}SrcBMaxFwdN_{num}': src_b_max_fwd_l,
        f'{prefix}SrcBMeanFwdN_{num}': src_b_mean_fwd_l,
        f'{prefix}SrcBStdFwdN_{num}': src_b_std_fwd_l,
        f'{prefix}SrcBSumBwdN_{num}': src_b_sum_bwd_l,
        f'{prefix}SrcBMinBwdN_{num}': src_b_min_bwd_l,
        f'{prefix}SrcBMaxBwdN_{num}': src_b_max_bwd_l,
        f'{prefix}SrcBMeanBwdN_{num}': src_b_mean_bwd_l,
        f'{prefix}SrcBStdBwdN_{num}': src_b_std_bwd_l,
        f'{prefix}Time2FlowFwdN_{num}': time_flow_fwd_sec_l,
        f'{prefix}Time2FlowBwdN_{num}': time_flow_bwd_sec_l
    })
    return pd.concat([f_df, new_features_df], axis=1)

def build_time_bet_2_flow_time_window(f_df, minutes):
    '''
        Given Dataframe and Minutes, Creates a new columns with
        the time between 2 flows in time window for both SrcAddr and DstAddr
    '''
    data_f = f_df[['SrcAddr', 'DstAddr', 'is_fwd', 'epoch']].copy()
    window_range_sec = minutes * 60

    src_fwd_dict = {}
    src_bwd_dict = {}
    dst_fwd_dict = {}
    dst_bwd_dict = {}

    #Time Between Flows
    src_fwd_list = []
    src_bwd_list = []

    dst_fwd_list = []
    dst_bwd_list = []

    for index, row in data_f.iterrows():
        print_status(index, len(data_f.index))
        if row.is_fwd:
            helper_trim_dict_value_list(row.SrcAddr, src_fwd_dict, 2)
            helper_trim_dict_value_list(row.DstAddr, dst_fwd_dict, 2)

            src_fwd_dict[row.SrcAddr].append(row.epoch)
            dst_fwd_dict[row.DstAddr].append(row.epoch)

            src_bwd_list.append(0)
            dst_bwd_list.append(0)

            #Forward Src
            src_fwd_list.append(helper_calc_flow_diff(src_fwd_dict[row.SrcAddr], window_range_sec))
            #Forward Dst
            dst_fwd_list.append(helper_calc_flow_diff(dst_fwd_dict[row.DstAddr], window_range_sec))
        else:
            helper_trim_dict_value_list(row.SrcAddr, src_bwd_dict, 2)
            helper_trim_dict_value_list(row.DstAddr, dst_bwd_dict, 2)

            src_bwd_dict[row.SrcAddr].append(row.epoch)
            dst_bwd_dict[row.DstAddr].append(row.epoch)

            src_fwd_list.append(0)
            dst_fwd_list.append(0)

            #Backward Src
            src_bwd_list.append(helper_calc_flow_diff(src_bwd_dict[row.SrcAddr], window_range_sec))
            #Backward Dst
            dst_bwd_list.append(helper_calc_flow_diff(dst_bwd_dict[row.DstAddr], window_range_sec))
    #Build Data Frame
    new_features_df = pd.DataFrame(data={
        f'STime2FlowFwd_{minutes}T': src_fwd_list,
        f'STime2FlowBwd_{minutes}T': src_bwd_list,
        f'DTime2FlowFwd_{minutes}T': dst_fwd_list,
        f'DTime2FlowBwd_{minutes}T': dst_bwd_list
    })
    return pd.concat([f_df, new_features_df], axis=1)

if __name__ == '__main__':
    main()
