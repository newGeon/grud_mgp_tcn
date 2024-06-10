import os
import pickle

from main_preprocessing_mgp_tcn_v2_0_1 import load_data

# MGP-TCN 데이터 전처리 코드만 실행하는 실행 코드
if __name__ == "__main__":
    """
    # 원본 데이터 생성 코드                    1_1_origin
    # Origin Data Creating Code             1_1_origin
    """
    print("MGP-TCN Data Preprocessing Code (Standard Data Create Code) -------------------------------")
    print("MGP-TCN 데이터 전처리 코드 실행 코드 정리 (기준 데이터 생성 코드) ---------------------------------")

    # list_kinds = ["ex1c", "whole1c"]
    list_kinds = ["ex1c"]

    na_thres = 500
    overwrite = 0
    data_sources = ["labs", "vitals", "covs"]
    hour = "55h"
    
    min_length = 7
    max_length = 200
    split = 0
    num_obs_thres = 10000

    # root_path = os.getcwd()
    root_path = "/home/cdss/mgp_tcn_origin"
    output_path = "/output_imputation/output_0/"

    print(root_path)
    # 원본 데이터 즉 기준 데이터 저장 로직
    experiment_type = "1_0_origin"

    for kind in list_kinds:
        for horizon in range(0, 8):
            print("kind = {}, horizon = {}".format(kind, horizon))

            data_path = root_path + output_path
            data_path += 'mgp-tcn-datadump_{}_{}_na_thres_{}_min_length_{}_max_length_{}_horizon_{}_split_{}.pkl'.format(hour, kind, na_thres, min_length, max_length, horizon, split)

            full_dataset = load_data(test_size=0.1, horizon=horizon, na_thres=na_thres, variable_start_index=5,
                                     data_sources=data_sources, min_length=min_length, max_length=max_length,
                                     split=split, binned=False, root_path=root_path, output_path=output_path,
                                     hour=hour, kind=kind)
            print('-----------------------------------------------------------------------------------------')
            print('horizon = ' + str(horizon) + ', kind = ' + kind + ' >>>>> pickle file make SUCESS -------')
            pickle.dump(full_dataset, open(data_path, "wb"))
            print('-----------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------------------')
