from common_parameter import *


def get_data_path(origin, ind):
    if origin == 'chinese':
        gt_path = path_preprocess_data + 'vcf/indian_origin/indian_origin_ground_truth.vcf'
        dv_path = path_preprocess_data + 'vcf/indian_origin/' + str(ind) + '_output.vcf'
        gt_csv_path = path_preprocess_data + 'csv/indian_origin/gt_labeled_' + str(ind) + '.csv'
        dv_csv_path = path_preprocess_data + 'csv/indian_origin/dv_labeled_' + str(ind) + '.csv'
        match_csv_path = path_preprocess_data + 'csv/indian_origin/matched_csv_' + str(ind) + '.csv'
        learning_csv_path = path_preprocess_data + 'csv/indian_origin/learning_csv_' + str(ind) + '.csv'
        result_csv_path = path_result_data + 'csv/indian_origin/result_csv_' + str(ind) + '.csv'
        confusion_csv_path = path_result_data + 'csv/indian_origin/confusion_' + str(ind) + '.csv'
        score_csv_path = path_result_data + 'csv/indian_origin/score_' + str(ind) + '.csv'
    else:
        gt_path = path_preprocess_data + 'vcf/chinese_origin/chinese_origin_ground_truth.vcf'
        dv_path = path_preprocess_data + 'vcf/chinese_origin/' + str(ind) + '_output.vcf'
        gt_csv_path = path_preprocess_data + 'csv/chinese_origin/gt_labeled_' + str(ind) + '.csv'
        dv_csv_path = path_preprocess_data + 'csv/chinese_origin/dv_labeled_' + str(ind) + '.csv'
        match_csv_path = path_preprocess_data + 'csv/chinese_origin/matched_csv_' + str(ind) + '.csv'
        learning_csv_path = path_preprocess_data + 'csv/chinese_origin/learning_csv_' + str(ind) + '.csv'
        result_csv_path = path_result_data + 'csv/chinese_origin/result_csv_' + str(ind) + '.csv'
        confusion_csv_path = path_result_data + 'csv/chinese_origin/confusion_' + str(ind) + '.csv'
        score_csv_path = path_result_data + 'csv/chinese_origin/score_' + str(ind) + '.csv'

    return gt_path, dv_path, gt_csv_path, dv_csv_path, match_csv_path,\
        learning_csv_path, result_csv_path, confusion_csv_path, score_csv_path

# vcf to csv for rhesus macaque individual
def read_vcf(path):
    with open(path, 'r') as f:
        lines_info = [l for l in tqdm(f) if not l.startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines_info)),
        dtype={'#CHROM': str, 'POS': str, 'ID': str, 'REF': str, 'ALT': str, 'QUAL': float, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})


# vcf to csv for rhesus macaque label
def read_vcf_(path, id=None):
    if id is not None:
        idx = ind_list_30.index(id) + 8
        with open(path, 'r') as f:
            lines_info = [l.split(' ')[0:8] + [l.split(' ')[idx]] for l in tqdm(f) if not l.startswith('#')]
        column_ = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "FORMAT", "default"]
        df_ = pd.DataFrame(lines_info, columns=column_)
        return df_
    else:
        with open(path, 'r') as f:
            lines_info = [l.split(' ') for l in tqdm(f) if not l.startswith('#')]
        column_ = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "FORMAT", "default"]
        df_ = pd.DataFrame(lines_info, columns=column_)
        return df_


def check_snp(ref, alt_set):
    if len(ref) > 1:
        return False

    for l_ in alt_set:
        if l_ != '*' and len(ref) != len(l_):
            return False
    else:
        return True


def return_type_(info, ref, alt):
    # print(info)
    if pd.isna(info):
        return 'WRONG'
    if len(info.split(':')[0].split('/')) == 1 or \
            (info.split(':')[0].split('/')[0] == '.' and info.split(':')[0].split('/')[1] == '.'):
        o1 = 'DROP'
        o2 = ''
    else:
        if info.split(':')[0].split('/')[0] == info.split(':')[0].split('/')[1]:
            o1 = 'hm'
        else:
            o1 = "ht"

        if len(ref) == 0:
            o2 = '.wrong'
        else:
            alt_set = set()
            for l_ in alt.split(','):
                alt_set.add(l_)

            if check_snp(ref, alt_set):
                o2 = '.snp'
            else:
                o2 = '.indel'
    if 'DROP' in o1 + o2 or 'wrong' in o1 + o2:
        return 'WRONG'
    else:
        return o1 + o2


# convert vcf to csv and labeling the variant types
def labeling_M(label_data, idx_number, gt_path, dv_path, gt_csv_path, dv_csv_path):
    if label_data == 'gt':
        label_path = gt_path
        csv_path = gt_csv_path

        vcf_df = read_vcf_(label_path, idx_number)
    else:
        label_path = dv_path
        csv_path = dv_csv_path

        vcf_df = read_vcf(label_path)
        vcf_df = vcf_df[vcf_df['CHROM'].isin(rhe_mac_chr_dict_10.keys())]
        vcf_df['CHROM'] = vcf_df['CHROM'].replace(rhe_mac_chr_dict_10)
        vcf_df = vcf_df.reset_index(drop=True)

    vcf_df['QUAL'] = vcf_df['QUAL'].astype(float)
    vcf_df['CHROM'] = vcf_df['CHROM'].astype(str)

    vcf_df = vcf_df[vcf_df['CHROM'].isin(chrom_values)]
    vcf_df = vcf_df.reset_index(drop=True)

    print(vcf_df)
    vcf_df['TYPE'] = ''
    labels = dict()
    for i in tqdm(range(len(vcf_df.index))):
        labels[i] = return_type_(vcf_df.loc[i, 'default'], vcf_df.loc[i, 'REF'], vcf_df.loc[i, 'ALT'])

    label_df = pd.DataFrame.from_dict(labels, orient='index', columns=['label'])
    vcf_df['TYPE'] = label_df['label']
    if 'INFO' in vcf_df.columns:
        vcf_df = vcf_df.drop('INFO', axis=1)
    vcf_df = vcf_df.loc[vcf_df['TYPE'] != 'WRONG']
    vcf_df = vcf_df.reset_index(drop=True)
    vcf_df.to_csv(csv_path)


# comparing and labeling the label between ground truth and rhesus macaque individuals
def match_L(gt_csv_path, dv_csv_path, match_csv_path):
    gt_df = pd.read_csv(gt_csv_path, index_col=0)
    print(gt_df)

    gt_dict = dict()
    for i in tqdm(range(len(gt_df.index))):
        gt_dict[str(gt_df.loc[i, 'CHROM']) + '_' + str(gt_df.loc[i, 'POS'])] = {'REF': gt_df.loc[i, 'REF'],
                                                                                'ALT': gt_df.loc[i, 'ALT'],
                                                                                'TYPE': gt_df.loc[i, 'TYPE'],
                                                                                'default': gt_df.loc[i, 'default']}

    count_dict = {'NOT_MATCH': 0, 'MATCH': 0, 'MATCH_BUT_WRONG': 0, 'MATCH_RIGHT_BUT_DIFF': 0, 'MATCH_ALL_RIGHT': 0,
                  'DIFFERENT_ALLELE': 0}
    labels = dict()
    dv_df = pd.read_csv(dv_csv_path, index_col=0)
    print(dv_df)
    for i in tqdm(range(len(dv_df.index))):
        key_ = str(dv_df.loc[i, 'CHROM']) + '_' + str(dv_df.loc[i, 'POS'])
        ref = dv_df.loc[i, 'REF']
        alt = dv_df.loc[i, 'ALT']
        _type = dv_df.loc[i, 'TYPE']
        fflag = None
        if key_ in gt_dict:
            ori_ref = gt_dict[key_]['REF']
            ori_ALT = gt_dict[key_]['ALT']
            ori_TYPE = gt_dict[key_]['TYPE']
            ori_INFO = gt_dict[key_]['default']
            count_dict['MATCH'] += 1
            if _type == ori_TYPE:
                if ref == ori_ref and (alt == ori_ALT or alt in ori_ALT):
                    count_dict['MATCH_ALL_RIGHT'] += 1
                    fflag = 'MATCH_ALL_RIGHT'
                else:
                    count_dict['MATCH_RIGHT_BUT_DIFF'] += 1
                    fflag = 'MATCH_RIGHT_BUT_DIFF'
                    if ref != ori_ref:
                        count_dict['DIFFERENT_ALLELE'] += 1
                        fflag += '_diff_allele'
            else:
                count_dict['MATCH_BUT_WRONG'] += 1
                fflag = 'MATCH_BUT_WRONG'
        else:
            count_dict['NOT_MATCH'] += 1
            ori_ref = None
            ori_ALT = None
            ori_TYPE = None
            ori_INFO = None
            fflag = 'NOT_MATCH'
        labels[i] = [ori_ref, ori_ALT, ori_TYPE, ori_INFO, fflag]

    print(count_dict)
    label_df = pd.DataFrame.from_dict(labels, orient='index',
                                      columns=['ORI_REF', 'ORI_ALT', 'ORI_TYPE', 'ORI_INFO', 'MATCH'])
    dv_df = dv_df.join(label_df[['ORI_REF', 'ORI_ALT', 'ORI_TYPE', 'ORI_INFO', 'MATCH']])
    dv_df.to_csv(match_csv_path)


# extract the features
def make_learning_data(data_csv, column_name, learning_csv_path):
    feature_column = []
    feature_column_dict = {}
    data_csv['QUAL'] = data_csv['QUAL'].astype(float)
    data_csv['FORMAT'] = data_csv['FORMAT'].astype(str)
    data_csv[column_name] = data_csv[column_name].astype(str)
    print('convert to learning data')
    for i in tqdm(range(len(data_csv.index))):
        # QUAL
        feature_column_dict[i] = {}
        feature_column_dict[i]['QUAL'] = data_csv.loc[i, 'QUAL']
        if 'QUAL' not in feature_column:
            feature_column.append('QUAL')

        # FORMAT
        for c, f in zip(data_csv.loc[i, 'FORMAT'].split(':')[1:], data_csv.loc[i, column_name].split(':')[1:]):
            feature_column_dict[i][c + '_mean'] = np.mean(eval(f))
            if c + '_mean' not in feature_column:
                feature_column.append(c + '_mean')

            feature_column_dict[i][c + '_diff'] = np.max(eval(f)) - np.min(eval(f))
            if c + '_diff' not in feature_column:
                feature_column.append(c + '_diff')

        # TYPE
        feature_column_dict[i]['TYPE'] = data_csv.loc[i, 'TYPE']

    input_csv = pd.DataFrame.from_dict(feature_column_dict, orient='index')
    input_csv = input_csv.reset_index(drop=True)
    input_csv.to_csv(learning_csv_path)
    return input_csv


# apply the GVRP refinement model
def apply_model(input_csv, data_csv, model_dir):
    print('\nmodel inference step')
    model_dict = {}
    feature_order = ['QUAL', 'AD_mean', 'AD_diff', 'DP_mean', 'DP_diff', 'GQ_mean', 'GQ_diff', 'PL_mean', 'PL_diff',
                     'VAF_mean', 'VAF_diff']  # 학습할 때 사용했던 순서
    types_ = ['hm.indel', 'ht.indel', 'hm.snp', 'ht.snp', 'WRONG']
    # m_df = pd.DataFrame()
    for t_ in types_:
        print(t_)
        data_type_ = input_csv.loc[input_csv['TYPE'] == t_]
        data_type_ = data_type_.drop('TYPE', axis=1)
        data_type_ = data_type_[feature_order]
        print(len(data_type_))
        print(data_type_)
        if len(data_type_) == 0:
            pass
        else:
            if t_ == 'WRONG':
                pred_list = [0] * len(data_type_)
                pred_p_list = [0] * len(data_type_)
            else:
                model = pickle.load(open(model_dir + t_ + '_XG_Boost_model.pkl', 'rb'))
                model_dict[t_] = model

                pred_list = model.predict(data_type_)
                pred_p_list = model.predict_proba(data_type_)[:, 1]
                print(t_, Counter(pred_list))

            data_csv.loc[data_csv['TYPE'] == t_, 'result'] = pred_list
            data_csv.loc[data_csv['TYPE'] == t_, 'result_p'] = pred_p_list


# apply the GVRP refinement model
def rhe_L(match_csv_path, learning_csv_path, result_csv_path, model_dir):
    match_csv = pd.read_csv(match_csv_path, index_col=0)
    match_csv['MATCH'] = match_csv['MATCH'].apply(lambda x: 1 if x == 'MATCH_ALL_RIGHT' else 0)
    print(match_csv['MATCH'].value_counts())
    print(match_csv)

    input_csv = make_learning_data(match_csv, 'default', learning_csv_path)
    print(input_csv)

    apply_model(input_csv, match_csv, model_dir)
    print(match_csv)
    print(match_csv['result'].value_counts())
    match_csv.to_csv(result_csv_path)


# calculate the confusion matrix
def return_metric(df_t):
    tp = len(df_t[df_t['confusion_M'] == 'TP'].reset_index(drop=True).index)
    tn = len(df_t[df_t['confusion_M'] == 'TN'].reset_index(drop=True).index)
    fp = len(df_t[df_t['confusion_M'] == 'FP'].reset_index(drop=True).index)
    fn = len(df_t[df_t['confusion_M'] == 'FN'].reset_index(drop=True).index)
    if tp == 0 and fp == 0:
        precision = None
    else:
        precision = round(tp / (tp + fp), 3)
    if tp == 0 and fn == 0:
        recall = None
    else:
        recall = round(tp / (tp + fn), 3)
    if tn == 0 and fp == 0:
        mis_filtering = None
    else:
        mis_filtering = round(tn / (tn + fp), 3)
    if tp == 0 and tn == 0 and fp == 0 and fn == 0:
        accuracy = None
    else:
        accuracy = round((tp + tn) / (tp + tn + fp + fn), 3)
    if precision == 0 or precision is None or recall == 0 or recall is None:
        f1_score = None
    else:
        f1_score = round(2 / (1 / precision + 1 / recall), 3)
    return tp, tn, fp, fn, precision, recall, f1_score, mis_filtering, accuracy


# save the result of GVRP refinement model appliance
def met(result_csv_path, confusion_csv_path, score_csv_path):
    result_df = pd.read_csv(result_csv_path, index_col=0)
    # result_df['MATCH'] = result_df['MATCH'].apply(lambda x: 1 if x=='MATCH_ALL_RIGHT' else 0)
    print(result_df)
    confusion_M = {}
    for i in tqdm(range(len(result_df.index))):
        if result_df.loc[i, 'MATCH'] == 0 and result_df.loc[i, 'result'] == 0:
            flag_ = 'TN'
        elif result_df.loc[i, 'MATCH'] == 0 and result_df.loc[i, 'result'] == 1:
            flag_ = 'FP'
        elif result_df.loc[i, 'MATCH'] == 1 and result_df.loc[i, 'result'] == 0:
            flag_ = 'FN'
        elif result_df.loc[i, 'MATCH'] == 1 and result_df.loc[i, 'result'] == 1:
            flag_ = 'TP'
        else:
            flag_ = None
        confusion_M[i] = flag_

    confusion_df = pd.DataFrame.from_dict(confusion_M, orient='index', columns=['confusion_M'])
    result_df['confusion_M'] = confusion_df['confusion_M']
    result_df.to_csv(confusion_csv_path)

    confusion_df = pd.read_csv(confusion_csv_path, index_col=0)
    print(confusion_df)
    type_list = ['ht.indel', 'ht.snp', 'hm.indel', 'hm.snp']
    column_list = ['TYPE', 'total_#', 'sum of all', '# of TP', '# of TN', '# of FP',
                   '# of FN', 'precision', 'recall', 'f1 score', 'filtering_rate', 'accuracy']
    result_list = []
    for type_ in type_list:
        df_t = confusion_df[confusion_df['TYPE'] == type_].reset_index(drop=True)
        tp, tn, fp, fn, precision, recall, f1_score, filtering, accuracy = return_metric(df_t)
        print('TYPE :', type_, 'total_# :', len(df_t.index), 'sum of all :', tp + tn + fp + fn,
              '# of TP :', tp, '# of TN :', tn, '# of FP :', fp, '# of FN :', fn,
              'precision :', precision, 'recall :', recall, 'f1 score :', f1_score,
              'filtering_rate :', filtering, 'accuracy :', accuracy)
        t_list = [type_, len(df_t.index), tp + tn + fp + fn, tp, tn, fp, fn, precision, recall, f1_score, filtering,accuracy]
        result_list.append(t_list)

    tp, tn, fp, fn, precision, recall, f1_score, filtering, accuracy = return_metric(confusion_df)
    print('TYPE : ALL', 'total_# :', len(confusion_df.index), 'sum of all :', tp + tn + fp + fn,
          '# of TP :', tp, '# of TN :', tn, '# of FP :', fp, '# of FN :', fn,
          'precision :', precision, 'recall :', recall, 'f1 score :', f1_score,
          'filtering_rate :', filtering, 'accuracy :', accuracy)
    t_list = ['ALL', len(confusion_df.index), tp + tn + fp + fn, tp, tn, fp, fn, precision, recall, f1_score, filtering, accuracy]
    result_list.append(t_list)

    confusion_df['TYPE'] = confusion_df['TYPE'].astype(str)
    df_indel = confusion_df[confusion_df['TYPE'].str.contains('.indel')].reset_index(drop=True)
    tp, tn, fp, fn, precision, recall, f1_score, filtering, accuracy = return_metric(df_indel)
    print('TYPE : Indels', 'total_# :', len(confusion_df.index), 'sum of all :', tp + tn + fp + fn,
          '# of TP :', tp, '# of TN :', tn, '# of FP :', fp, '# of FN :', fn,
          'precision :', precision, 'recall :', recall, 'f1 score :', f1_score,
          'filtering_rate :', filtering, 'accuracy :', accuracy)
    t_list = ['Indels', len(confusion_df.index), tp + tn + fp + fn, tp, tn, fp, fn, precision, recall, f1_score, filtering, accuracy]
    result_list.append(t_list)

    confusion_df['TYPE'] = confusion_df['TYPE'].astype(str)
    df_snp = confusion_df[confusion_df['TYPE'].str.contains('.snp')].reset_index(drop=True)
    tp, tn, fp, fn, precision, recall, f1_score, filtering, accuracy = return_metric(df_snp)
    print('TYPE : SNPs', 'total_# :', len(confusion_df.index), 'sum of all :', tp + tn + fp + fn,
          '# of TP :', tp, '# of TN :', tn, '# of FP :', fp, '# of FN :', fn,
          'precision :', precision, 'recall :', recall, 'f1 score :', f1_score,
          'filtering_rate :', filtering, 'accuracy :', accuracy)
    t_list = ['SNPs', len(confusion_df.index), tp + tn + fp + fn, tp, tn, fp, fn, precision, recall, f1_score, filtering, accuracy]
    result_list.append(t_list)

    df_r = pd.DataFrame(result_list, columns=column_list)
    df_r.to_csv(score_csv_path)
