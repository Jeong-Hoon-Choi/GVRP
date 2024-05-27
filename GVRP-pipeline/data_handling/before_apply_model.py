from GVRP_lib import *


def check_snp(ref, alt_set):
    if len(ref) > 1:
        return False

    for l_ in alt_set:
        if l_ != '*' and len(ref) != len(l_):
            return False
    else:
        return True


def return_type_(info, ref, alt):
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


def labeling_df(vcf_df, column_name):
    vcf_df['TYPE'] = ''
    labels = dict()
    for i in tqdm(range(len(vcf_df.index))):
        labels[i] = return_type_(vcf_df.loc[i, column_name], vcf_df.loc[i, 'REF'], vcf_df.loc[i, 'ALT'])

    label_df = pd.DataFrame.from_dict(labels, orient='index', columns=['label'])
    vcf_df['TYPE'] = label_df['label']
    if 'INFO' in vcf_df.columns:
        vcf_df = vcf_df.drop('INFO', axis=1)
    vcf_df = vcf_df.loc[vcf_df['TYPE'] != 'WRONG']
    vcf_df = vcf_df.reset_index(drop=True)
    return vcf_df


def make_learning_data(data_csv, column_name):
    feature_column = []
    feature_column_dict = {}
    data_csv['QUAL'] = data_csv['QUAL'].astype(float)
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
    return input_csv