from preprocess_for_GVRP_rhesus_macaque import *


if __name__ == '__main__':
    for ind_list in ind_list_all:
        if ind_list == ind_list_china:
            origin = 'Chinese'
        else:
            origin = 'Indian'
        for ind in ind_list:
            print(ind)

            gt_path, dv_path, gt_csv_path, dv_csv_path, match_csv_path, \
                learning_csv_path, result_csv_path, confusion_csv_path, score_csv_path = get_data_path(origin, ind)

            labeling_M('gt', ind, gt_path, dv_path, gt_csv_path, dv_csv_path)
            labeling_M('dv', ind, gt_path, dv_path, gt_csv_path, dv_csv_path)
            match_L(gt_csv_path, dv_csv_path, match_csv_path)
            rhe_L(match_csv_path, learning_csv_path, result_csv_path, model_dir)
            met(result_csv_path, confusion_csv_path, score_csv_path)
