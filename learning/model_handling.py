from GVRP_lib import *


def apply_model(input_csv, data_csv, model_dir):
    feature_order = ['AD_diff', 'DP_mean', 'GQ_mean', 'PL_diff', 'VAF_mean', 'QUAL', 'M_ratio', 'S_ratio',
                     'Total_reads', 'mean_mapq', 'low_mapq_ratio', 'forward_strand_ratio'] # trained feature order
    input_csv = input_csv[feature_order]
    model = pickle.load(open(model_dir, 'rb'))
    pred_list = model.predict(input_csv)
    data_csv.loc['result'] = pred_list
