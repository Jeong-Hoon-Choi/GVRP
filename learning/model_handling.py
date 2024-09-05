from GVRP_lib import *


def apply_model(input_csv, data_csv, model_dir):
    model_dict = {}
    feature_order = ['QUAL', 'AD_mean', 'AD_diff', 'DP_mean', 'DP_diff', 'GQ_mean', 'GQ_diff', 'PL_mean', 'PL_diff',
                     'VAF_mean', 'VAF_diff']  # trained feature order
    types_ = ['hm.indel', 'ht.indel', 'hm.snp', 'ht.snp', 'WRONG']
    for t_ in types_:
        data_type_ = input_csv.loc[input_csv['TYPE'] == t_]
        data_type_ = data_type_.drop('TYPE', axis=1)
        data_type_ = data_type_[feature_order]

        if len(data_type_) == 0:
            pass
        else:
            if t_ == 'WRONG':
                pred_list = [0] * len(data_type_)
            else:
                model = pickle.load(open(model_dir + t_ + '_model.pkl', 'rb'))
                model_dict[t_] = model

                pred_list = model.predict(data_type_)

            data_csv.loc[data_csv['TYPE'] == t_, 'result'] = pred_list