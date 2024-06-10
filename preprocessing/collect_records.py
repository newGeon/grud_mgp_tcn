'''
---------------------------------
Author: Michael Moor, 13.03.2018
---------------------------------

Collect records that are spread over different rows in input file such that
    1) records of same icustay_id and chart_time are on same row
    2) redundant values for one id and time are averaged over

This script is able to process a varying number of medical variables assuming that they are listed in columns after (or instead of) the other three variables.

collect_records(file_in_path, file_out_path):
    input: path to file to be processed
    output: processed file at file_out_path   
'''
import os
import sys
import csv
import time
from datetime import timedelta
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.linear_model import LinearRegression

warnings.filterwarnings(action='ignore')


def collect_records(filepath=None, outpath=None):

    print("Collecting extracted database records in single lines per observation time")
    start = time.time() # Get current time

    # -----------------------------------------------------------------------------
    #Step 1. Open the input file:
    try:
        f_in = open(filepath, 'r')
    except IOError:
        print("Cannot read input file %s" % filepath)
        sys.exit(1)


    # -----------------------------------------------------------------------------
    #Step 2. read file line by line

    #first read header by reading first line
    line = f_in.readline()

    #print('First Line: {}'.format(line))

    parts = line.rstrip().split(",") # split the line

    firstrow2write = parts # will be written to the output file as first line

    #print('First parts: {}'.format(parts))

    variable_indices = range(4, len(parts))
    variables = parts[4: len(parts)] #containing the medical variable names
    var_counter = np.zeros(len(variable_indices)) #counting how many observation for certain timepoint available (for averaging over)
    var_sum = np.repeat(np.nan, len(variable_indices)) #summing the value of all variables for certain timepoint

    #process first line of values for initializing:
    line = f_in.readline()
    parts = line.rstrip().split(",")
    current_icustay = parts[0]
    current_time = parts[2]
    header = parts[0:4]

    tmp_values = parts[4:len(parts)] # get list of all medical values as each as string

    current_values = np.repeat(np.nan, len(variable_indices)) # initialize current values to NANs

    # Process each value of the first line:
    for i in range(len(current_values)): # convert only available numbers as integer to current_values array
        if tmp_values[i] != '':
            current_values[i] = float(tmp_values[i])
            var_counter[i] += 1 #for each non-NAN value count it for each variable seperately
            if var_sum[i] != var_sum[i]: #check if it is a nan
                var_sum[i] = current_values[i] #set it to the new value, as np.nans stay nan when adding numbers
            else:
                var_sum[i] += current_values[i] # sum all observations of each variable up (seperately) to later build timepoint-wise average

    # Open the output file
    with open(outpath, 'w') as data_file: 
        data_writer = csv.writer(data_file, delimiter=',')

        data_writer.writerow(firstrow2write) # Write header information to first line of outfile
        
        #process the remaining lines:
        for line in f_in:
            # Strip of the \n and split the line
            parts = line.rstrip().split(",")
            # Get new id
            new_icustay = parts[0]
            new_time = parts[2]
            new_header = parts[0:4]

            # Check if patient or point in time have changed! if yes, compute average of the medical variables and write to outfile
            if (new_icustay != current_icustay) or (new_time != current_time):
                # for each entry of var_sum divide by var_counter iff number available (non-NAN)
                averages = np.repeat(np.nan, len(var_sum))
    
                for i in range(len(var_sum)):
                    if var_sum[i] == var_sum[i]: # if var_sum is NOT a NaN, compute average
                        averages[i] = var_sum[i]/var_counter[i]
                # write this array of averages (and potentially NANs) to a line of output file
                row2write = np.append(header, averages)
                data_writer.writerow(row2write)
                #f_out.write("%s,%s,%s\n" % (header ...))

                # reinitialise icustay_id, time and header such that next timepoint (and or patient) can be processed.
                current_icustay = new_icustay
                current_time = new_time
                header = new_header
                # reinitialise count and sum of variables for computing new averages:
                var_counter = np.zeros(len(variable_indices)) #counting how many observation for certain timepoint available (for averaging over)
                var_sum = np.repeat(np.nan, len(variable_indices)) # summing over the variables for certain timepoint (numerator for average)

        	# Process the current (patient, time) tuple!	
            tmp_values = parts[4:len(parts)]

            new_values = np.repeat(np.nan, len(variable_indices)) # initialize current values to NANs

            # Process each value of the line:
            for i in range(len(new_values)): # convert only available numbers to integer to current_values array
                if tmp_values[i] != '':
                    new_values[i] = float(tmp_values[i])
                    var_counter[i] += 1 #for each non-NAN value count it for each variable seperately
                    if var_sum[i] != var_sum[i]: #check if it is a nan
                        var_sum[i] = new_values[i] #set it to the new value, as np.nans stay nan when adding numbers
                    else:
                        var_sum[i] += new_values[i] # sum all observations of each variable up (seperately) to later build timepoint-wise average
    # Close the files
    f_in.close()

    end = time.time()
    print('Collecting records RUNTIME {} seconds'.format(end - start)) # Print runtime of this process

    return None

# 모든 데이터가 null인 경우 컬럼 삭제
def fn_null_column_delete(df_0_case, df_0_control):

    row_cnt = df_0_case.shape[0]
    list_parameter = df_0_case.columns[4:]
    list_null = []
    
    df_null = df_0_case.isnull().sum()

    for parameter in list_parameter:
        if df_null[parameter] == row_cnt:
            list_null.append(parameter)
            print('ALL NULL >> ' + parameter + ' : row_cnt = ' + str(row_cnt) + ', cnt = ' + str(df_null[parameter]))

    df_1_case = df_0_case.drop(list_null, axis=1)
    df_1_control = df_0_control.drop(list_null, axis=1)

    return df_1_case, df_1_control


# 선형회귀를 활용한 데이터 결측치 보강 코드
def fn_linear_regression_fill(df_0_case, df_0_control):

    # case 데이터를 기준으로 독립변수, 종속변수 설정
    row_cnt = df_0_case.shape[0]
    list_parameter = df_0_case.columns[4:]
    list_independent = []       # 독립변수
    list_dependent = []         # 종속변수
    list_none = []              # 증강 못하는 변수 (데이터가 너무 적은 경우, 50개 미만)
    
    df_null_case = df_0_case.isnull().sum()
    df_null_control = df_0_control.isnull().sum()

    for parameter in list_parameter:
        if (row_cnt - df_null_case[parameter]) >= 12000:
            # 12,000개가 넘는 경우 독립변수로 설정
            list_independent.append(parameter)
            print('독립변수 >> ' + parameter + ' : row_cnt = ' + str(row_cnt) + ', cnt = ' + str(row_cnt - df_null_case[parameter]))
        elif (row_cnt - df_null_case[parameter]) >= 50:
            # 50 ~ 12,0000개 사이면 종속변수로 설정  
            list_dependent.append(parameter)
            print('종속변수 >> ' + parameter + ' : row_cnt = ' + str(row_cnt) + ', cnt = ' + str(row_cnt - df_null_case[parameter]))
        else:
            list_none.append(parameter)
            print('증강 못하는 변수 >> ' + parameter + ' : row_cnt = ' + str(row_cnt) + ', cnt = ' + str(row_cnt - df_null_case[parameter]))

    for column in list_independent:
        print(column)
        list_case = df_0_case[df_0_case[column].notnull()][column].to_list()
        list_control = df_0_control[df_0_control[column].notnull()][column].to_list()

        # case 군        
        mean_case = np.mean(list_case)       # 평균
        std_case = np.std(list_case)         # 표준편차

        null_case_cnt = df_null_case[column]
        random_samples_case = np.random.normal(mean_case, std_case, null_case_cnt)

        null_data_case = df_0_case[column].isnull()  # 결측치 있는 데이터에 대한 mask
        df_0_case.loc[null_data_case, column] = random_samples_case

        # control 군
        mean_control = np.mean(list_control)       # 평균
        std_control = np.std(list_case)         # 표준편차

        null_control_cnt = df_null_control[column]
        random_samples_control = np.random.normal(mean_control, std_control, null_control_cnt)

        null_data_control = df_0_control[column].isnull()  # 결측치 있는 데이터에 대한 mask
        df_0_control.loc[null_data_control, column] = random_samples_control
    
    for column in list_dependent:
        new_list = list_independent.copy()
        new_list.append(column)

        # case군
        df_set_case = df_0_case.dropna(subset=[column])

        # 독립변수와 종속변수로 데이터 분할
        X_case = df_set_case[list_independent]
        y_case = df_set_case[column]

        # 회귀모델 학습
        model_case = LinearRegression()
        model_case.fit(X_case, y_case)

        # 결측치가 있는 행 선택
        missing_case_data = df_0_case[df_0_case[column].isnull()]
        missing_case_data = missing_case_data[new_list]

        # 예측을 통해 결측치 채우기
        predicted_case_values = model_case.predict(missing_case_data[list_independent])

        # 마이너스 값을 제외하고 결측치 채우기
        predicted_case_values = np.where(predicted_case_values < 0, np.nan, predicted_case_values)

        df_0_case.loc[df_0_case[column].isnull(), column] = predicted_case_values

        # control군
        df_set_control = df_0_control.dropna(subset=[column])

        # 독립변수와 종속변수로 데이터 분할
        X_control = df_set_control[list_independent]
        y_control = df_set_control[column]

        # 회귀모델 학습
        model_control = LinearRegression()
        model_control.fit(X_control, y_control)

        # 결측치가 있는 행 선택
        missing_control_data = df_0_control[df_0_control[column].isnull()]
        missing_control_data = missing_control_data[new_list]

        # 예측을 통해 결측치 채우기
        predicted_control_values = model_control.predict(missing_control_data[list_independent])

        # 마이너스 값을 제외하고 결측치 채우기
        predicted_control_values = np.where(predicted_control_values < 0, np.nan, predicted_control_values)

        df_0_control.loc[df_0_control[column].isnull(), column] = predicted_control_values

    return df_0_case, df_0_control


# GRU-D 알고리즘 적용 데이터 결측치 보강
def fn_grud_imputaion(df_data):
    list_icuid = df_data['icustay_id'].unique()
    
    list_parameters = list(df_data.columns[4:])
    list_mean = df_data.iloc[:,4:].mean()
    
    # mask 데이터를 먼저 계산
    df_mask = df_data.copy()
    for param in list_parameters:
        df_mask[param] = df_mask[param].apply(lambda x: 0 if np.isnan(x) else 1)
    
    # delta 데이터프레임 선언
    df_delta = pd.DataFrame()
    
    # 저장 데이터프레임 선언
    df_grud = pd.DataFrame()
    
    for icuid in tqdm(list_icuid):
        df_0_x = df_data.query('icustay_id == @icuid')
        df_0_x = df_0_x.reset_index()

        df_0_save = df_0_x.copy()

        # 시간 계산
        df_0_x['chart_time'] = pd.to_datetime(df_0_x['chart_time'])
        first_chart_time = df_0_x.loc[0, 'chart_time']
        
        df_0_x['chart_time'] = (df_0_x['chart_time'] - first_chart_time) / timedelta(days=1)

        # mask 데이터 추출
        df_0_mask = df_mask.query('icustay_id == @icuid')
        df_0_mask = df_0_mask.reset_index()

        df_0_delta = df_0_x.copy()
        df_0_delta = df_0_delta.reset_index()

        # delta 계산
        for param in list_parameters:
            for idx in range(len(df_0_delta[param])):
                if idx == 0:
                    # 첫번째 인덱스는 무조건 0으로 셋팅
                    df_0_delta[param][idx] = 0

                elif df_0_mask[param][idx] == 0:
                    # mask 값이 0인 경우
                    df_0_delta[param][idx] = df_0_delta['chart_time'][idx] - df_0_delta['chart_time'][idx-1] + df_0_delta[param][idx-1]
                else:
                    # mask 값이 1인 경우
                    df_0_delta[param][idx] = df_0_delta['chart_time'][idx] - df_0_delta['chart_time'][idx-1]

        df_delta = pd.concat([df_delta, df_0_delta])

        # GRU-D 알고리즘을 활용한 데이터 결측치 보강
        for param in list_parameters:
            start_idx = 0
            v_mean = list_mean[param]
            
            # 최초 값이 있는 시간 부터 데이터 결측치 보강
            for idx in range(len(df_0_mask[param])):
                if df_0_mask[param][idx] == 1:
                    start_idx = idx
                    break

            for idx in range(start_idx+1, len(df_0_mask[param])):                
                
                v_mask = df_0_mask[param][idx]
                
                if v_mask == 0:                
                    v_prev_x = df_0_save[param][idx - 1]                
                    v_delta = df_0_delta[param][idx]
                
                    df_0_save[param][idx] = v_delta * v_prev_x + (1 - v_delta) * v_mean

        df_grud = pd.concat([df_grud, df_0_save])

    return df_grud, df_delta


# 임상실험 관점으로 데이터 결측치 보강
def fn_clinical_fill(df_0_data, file_path=None):
    
    start = time.time() # Get current time

    # df_0_data = pd.read_csv(file_path)
    df_total = pd.DataFrame(columns=df_0_data.columns)

    list_icuid = df_0_data['icustay_id'].unique()

    for icuid in list_icuid:
        te_df = df_0_data.query('icustay_id == @icuid')
        te_df = te_df.fillna(method='ffill')

        df_total = pd.concat([df_total, te_df])

    df_total.to_csv(file_path, index=False, encoding='UTF-8') 

    end = time.time()
    print('DATA augmentaion for clinical way ----- SUCESS : {} seconds -----'.format(end - start))

    return df_total

# SNR 데이터 증강 중 static 데이터 증강
def fn_snr_augmentation_static(df_static):

    df_static_1 = df_static.copy()
    df_static_2 = df_static.copy()

    df_static_1['icustay_id'] = df_static_1['icustay_id'] + 300000
    df_static_2['icustay_id'] = df_static_2['icustay_id'] + 600000

    df_save_static = pd.concat([df_static, df_static_1, df_static_2], ignore_index=True)
    return df_save_static


# SNR 데이터 증강 중 labsvitals 데이터 증강
def fn_snr_augmentation_labvitals(df_data, snr_value):
    
    alpha = 1 / snr_value
    
    variable_colums = df_data.columns[4:]

    df_data['random_normal_1'] = np.random.standard_normal(df_data.shape[0])
    df_data['random_normal_2'] = np.random.triangular(-1, 0, 1, df_data.shape[0])

    df_augment_1 = df_data.copy()
    df_augment_2 = df_data.copy()

    for column in variable_colums:
        df_augment_1[column] = np.where(df_augment_1[column].isna(), df_augment_1[column], df_augment_1[column] + df_augment_1[column] * alpha * df_augment_1['random_normal_1'])
        df_augment_2[column] = np.where(df_augment_2[column].isna(), df_augment_2[column], df_augment_2[column] + df_augment_2[column] * alpha * df_augment_2['random_normal_2'])

    df_augment_1['icustay_id'] = df_augment_1['icustay_id'] + 300000
    df_augment_2['icustay_id'] = df_augment_2['icustay_id'] + 600000

    df_save = pd.concat([df_data, df_augment_1, df_augment_2], ignore_index=True)
    df_save = df_save.drop(columns=['random_normal_1', 'random_normal_2'])

    return df_save

