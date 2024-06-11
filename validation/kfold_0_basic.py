import pickle
import numpy as np
import collections


def fn_create_kfold_data(horizon, kfold, term_case, term_control, add_cnt, end_case, start_control, file_path, save_path):
    # 파일 이름
    file_name = "mgp-tcn-datadump_55h_ex1c_na_thres_500_min_length_7_max_length_200_horizon_{}_split_0.pkl".format(horizon)
    save_name = "mgp-tcn-datadump_55h_horizon_{}_kfold_{}.pkl".format(horizon, kfold)

    print("Loading existing preprocessed data dump ------------------------------------------")
    full_dataset = pickle.load(open(file_path + file_name, "rb"))

    # 첫번째 full_dataset은 Kfold = 0로 저장
    pickle.dump(full_dataset, open(save_path + save_name, "wb"))

    label_0_train = full_dataset[1][4]
    label_1_validation = full_dataset[2][4]
    label_2_test = full_dataset[3][4]

    print("----------------------------------------------------------------------------------")
    print(len(full_dataset[0]))
    print(full_dataset[0])
    print("----------------------------------------------------------------------------------")
    print(collections.Counter(label_0_train)[1])
    print(collections.Counter(label_0_train)[0])
    print("----------------------------------------------------------------------------------")
    print(collections.Counter(label_1_validation)[1])
    print(collections.Counter(label_1_validation)[0])
    print("----------------------------------------------------------------------------------")
    print(collections.Counter(label_2_test)[1])
    print(collections.Counter(label_2_test)[0])

    for i in range(0, 8, 2):
        kfold = int(i / 2 + 1)
        # (Train : Validation : Test) = (8 : 1 : 1)
        new_full_dataset = []
        new_full_dataset.append(full_dataset[0])

        if i == 6:
            # index 초과
            print(i)
            idx_val_sta_case = term_case * i
            idx_val_end_case = term_case * (i + 1)

            idx_tst_sta_case = term_case * (i + 1)
            idx_tst_end_case = end_case
        else:
            idx_val_sta_case = term_case * i
            idx_val_end_case = term_case * (i + 1)

            idx_tst_sta_case = term_case * (i + 1)
            idx_tst_end_case = term_case * (i + 2)

        idx_val_sta_ctol = start_control + term_control * i
        idx_val_end_ctol = start_control + term_control * (i + 1)

        idx_tst_sta_ctol = start_control + term_control * (i + 1)
        idx_tst_end_ctol = start_control + term_control * (i + 2)

        print("--- {} --------------------------------------".format(i))
        print("case : start_val_idx = {}, end_val_indx = {}".format(idx_val_sta_case, idx_val_end_case))
        print("case : start_tst_idx = {}, end_tst_indx = {}".format(idx_tst_sta_case, idx_tst_end_case))

        print("control : start_val_idx = {}, end_val_indx = {}".format(idx_val_sta_ctol, idx_val_end_ctol))
        print("control : start_tst_idx = {}, end_tst_indx = {}".format(idx_tst_sta_ctol, idx_tst_end_ctol))

        list_all_train = []
        list_all_validation = []
        list_all_test = []

        for i_i in range(10):
            temp_val_case = full_dataset[1][i_i][idx_val_sta_case:idx_val_end_case]
            temp_tst_case = full_dataset[1][i_i][idx_tst_sta_case:idx_tst_end_case]

            if i == 6:
                # 마지막 일 때만 temp_tst_case 추가
                temp_tst_6_case = full_dataset[2][i_i][0:add_cnt]
                temp_tst_case = np.concatenate((temp_tst_case, temp_tst_6_case))

            temp_val_control = full_dataset[1][i_i][idx_val_sta_ctol:idx_val_end_ctol]
            temp_tst_control = full_dataset[1][i_i][idx_tst_sta_ctol:idx_tst_end_ctol]

            dataset_train = np.delete(full_dataset[1][i_i], np.s_[idx_val_sta_ctol:idx_tst_end_ctol])
            dataset_train = np.delete(dataset_train, np.s_[idx_val_sta_case:idx_tst_end_case])

            if i == 6:
                dataset_train = np.concatenate((dataset_train, full_dataset[2][i_i][add_cnt:], full_dataset[3][i_i]))
            else:
                dataset_train = np.concatenate((dataset_train, full_dataset[2][i_i], full_dataset[3][i_i]))

            dataset_validation = np.concatenate((temp_val_case, temp_val_control))
            dataset_test = np.concatenate((temp_tst_case, temp_tst_control))

            list_all_train.append(dataset_train)
            list_all_validation.append(dataset_validation)
            list_all_test.append(dataset_test)

        # static 데이터
        temp_val_static_case = full_dataset[4][idx_val_sta_case:idx_val_end_case]
        temp_tst_static_case = full_dataset[4][idx_tst_sta_case:idx_tst_end_case]

        if i == 6:
            # 마지막 일 때만 temp_tst_static_case 추가
            temp_tst_static_6_case = full_dataset[5][0:add_cnt]
            temp_tst_static_case = np.concatenate((temp_tst_static_case, temp_tst_static_6_case), axis=0)

        temp_val_static_control = full_dataset[4][idx_val_sta_ctol:idx_val_end_ctol]
        temp_tst_static_control = full_dataset[4][idx_tst_sta_ctol:idx_tst_end_ctol]

        static_train = np.delete(full_dataset[4], np.s_[idx_val_sta_ctol:idx_tst_end_ctol], axis=0)
        static_train = np.delete(static_train, np.s_[idx_val_sta_case:idx_tst_end_case], axis=0)

        if i == 6:
            static_train = np.concatenate((static_train, full_dataset[5][add_cnt:], full_dataset[6]), axis=0)
        else:
            static_train = np.concatenate((static_train, full_dataset[5], full_dataset[6]), axis=0)

        static_validation = np.concatenate((temp_val_static_case, temp_val_static_control), axis=0)
        static_test = np.concatenate((temp_tst_static_case, temp_tst_static_control), axis=0)

        new_full_dataset.append(list_all_train)
        new_full_dataset.append(list_all_validation)
        new_full_dataset.append(list_all_test)

        new_full_dataset.append(static_train)
        new_full_dataset.append(static_validation)
        new_full_dataset.append(static_test)

        # pickle 저장
        save_name = "mgp-tcn-datadump_55h_horizon_{}_kfold_{}.pkl".format(horizon, kfold)
        pickle.dump(new_full_dataset, open(save_path + save_name, "wb"))

        """
        temp_case_00 = full_dataset[1][0][idx_start_case:idx_end_case]
        temp_control_00 = full_dataset[1][0][idx_start_control:idx_end_control]

        delete_00 = np.delete(full_dataset[1][0], np.s_[idx_start_control:idx_end_control])
        delete_00 = np.delete(full_dataset[1][0], np.s_[idx_start_case:idx_end_case])

        new_00 = np.concatenate((delete_00, full_dataset[2][0], full_dataset[3][0]))

        print("-------------------------------------------------------------------------------------------")
        """


if __name__ == "__main__":
    # 데이터 불균형으로 kfold 관련 기존 함수 사용이 어려워서 강제적으로 kfold 적용 코드 생성을 통한 데이터 분할
    list_horizon = [0, 1, 2, 3, 4, 5, 6, 7]
    list_term_case = [62, 62, 62, 62, 62, 61, 53, 49]
    list_term_control = [558, 558, 558, 557, 5556, 553, 530, 490]

    list_add_cnt = [50, 50, 50, 50, 51, 45, 0, 2]
    list_end_case = [446, 446, 446, 446, 445, 443, 424, 390]
    list_start_control = [450, 450, 450, 450, 450, 450, 440, 400]

    # path
    path_file = "../output_imputation/output_0/"
    path_save = "../output_imputation/output_0_kfold_5/"

    d_kfold = 0

    for i in range(0, 8):
        print("i = {}".format(i))
        d_horizon = list_horizon[i]
        d_term_case = list_term_case[i]
        d_term_control = list_term_control[i]
        d_add_cnt = list_add_cnt[i]
        d_end_case = list_end_case[i]
        d_start_control = list_start_control[i]

        print("horizon = {}".format(d_horizon))
        print("term_case = {}".format(d_term_case))
        print("term_control = {}".format(d_term_control))
        print("add_cnt = {}".format(d_add_cnt))
        print("end_case = {}".format(d_end_case))
        print("start_control = {}".format(d_start_control))
        print("-------------------------------------------------------------------------")

        fn_create_kfold_data(d_horizon, d_kfold, d_term_case, d_term_control, d_add_cnt, d_end_case, d_start_control, path_file, path_save)

    print("-------------------------------------------------------------------------------------------")
    print("--- SUCCESS -------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------------")
