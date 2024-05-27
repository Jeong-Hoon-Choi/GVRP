from common_parameter import *


if __name__ == '__main__':
    ind_list_all = [ind_list_china, ind_list_30]

    test_y_t = []
    pred_y_t = []
    roc_dict = {}

    for v_type in ['ht', 'hm']:
        if v_type == 'hm':
            v_l = 'HM-SNPs'
        else:
            v_l = 'HT-SNPs'
        print(v_type)
        test_y = []
        pred_y = []
        for ind_list in ind_list_all:
            if ind_list == ind_list_china:
                path_rsult_csv = path_result_data + 'csv/chinese_origin/'
                ind_tag = 'MMUL.CH-'
            else:
                path_rsult_csv = path_result_data + '/csv/indian_origin/'
                ind_tag = 'MMUL.IN-'
            for ind in ind_list:
                print(ind)
                ind_tag_ = ind_tag + str(ind)
                result_csv_p = path_rsult_csv + 'result_csv_' + str(ind) + '.csv'

                r_df = pd.read_csv(result_csv_p, index_col=0)
                r_df_type = r_df[r_df['TYPE'].str.contains(v_type + '.snp')].reset_index(drop=True)
                r_df_type['MATCH'] = r_df_type['MATCH'].apply(lambda x: 1 if x == 'MATCH_ALL_RIGHT' else 0)

                r_df_type['result_p'] = r_df_type['result_p'].astype(float)
                r_df_type['MATCH'] = r_df_type['MATCH'].astype(float)
                test_y_ind = r_df_type['MATCH'].tolist()
                pred_y_ind = r_df_type['result_p'].tolist()

                test_y += test_y_ind
                pred_y += pred_y_ind
                test_y_t += test_y_ind
                pred_y_t += pred_y_ind

                fpr, tpr, threshold_ = roc_curve(test_y_ind, pred_y_ind)
                auc_ = auc(fpr, tpr)

        fpr, tpr, threshold_ = roc_curve(test_y, pred_y)
        auc_ = auc(fpr, tpr)

        roc_dict[v_l] = {'fpr': fpr, 'tpr': tpr, 'auc_': auc_}

    fpr, tpr, threshold_ = roc_curve(test_y_t, pred_y_t)
    auc_ = auc(fpr, tpr)

    roc_dict['Total SNPs'] = {'fpr': fpr, 'tpr': tpr, 'auc_': auc_}

    plt.figure(figsize=(8, 8))
    for v_l in roc_dict:
        plt.plot(roc_dict[v_l]['fpr'], roc_dict[v_l]['tpr'], lw=2, label=v_l + ' (AUC = %0.3f)' % roc_dict[v_l]['auc_'])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve for Each Variant Type at Rhesus Macaque', fontsize=18)
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig(roc_save_dir + 'Rhesus_roc_curve_total.png', dpi=600)
