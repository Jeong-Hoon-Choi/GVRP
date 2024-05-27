from common_module import *


# vcf to csv
def read_vcf(path):
    with open(path, 'r') as f:
        lines_info = [l for l in tqdm(f) if not l.startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines_info)),
        dtype={'#CHROM': str, 'POS': str, 'ID': str, 'REF': str, 'ALT': str, 'QUAL': float, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})


# setting csv path for preprocessing data
def get_parameters(data_dir, output_dir, target_data, target_type):
    if target_data == 'HG001':
        label_path = data_dir + 'HG001_005-6_dedup_WGS.deepvar.giab_benchmark.GRCh38_concordance.vcf'
        data_path = data_dir + 'HG001_005-6_dedup_WGS.deepvar.GRCh38.PASS.' + target_type + '.vcf'
        learn_data_path = output_dir + target_data + target_type + '.csv'
        column_name = 'HG001_005'

    else:
        label_path = data_dir + 'HG002.giab_0028-9_dedup_WGS.deepvar.giab_benchmark.GRCh38_concordance.vcf'
        data_path = data_dir + 'HG002.giab_0028-9_WGS.GRCh38.deepvar.PASS.' + target_type + '.vcf'
        learn_data_path = output_dir + target_data + target_type + '.csv'
        column_name = 'HG002.giab_0029'

    return label_path, data_path, learn_data_path, column_name


# labeling the human variant based on GIAB data
def labeling_vcf(label_path, data_path):
    true_list = ['CONC_ST=TP', 'CONC_ST=TP,TP', 'CONC_ST=TP,TN', 'CONC_ST=TN,TP']
    non_list = ['CONC_ST=EMPTY']
    label_key_dict = {}
    vcf_df = read_vcf(label_path)
    print('convert vcf file to csv file')
    for i in tqdm(range(len(vcf_df.index))):
        key_ = str(vcf_df.loc[i, 'CHROM']) + '_' + str(vcf_df.loc[i, 'POS'])
        if vcf_df.loc[i, 'INFO'] in non_list:
            label_key_dict[key_] = 'NONE'
        elif vcf_df.loc[i, 'INFO'] in true_list:
            label_key_dict[key_] = 1
        else:
            label_key_dict[key_] = 0

    label_dict = {}
    data_df = read_vcf(data_path)
    print('labeling data')
    for i in tqdm(range(len(data_df.index))):
        key_ = str(data_df.loc[i, 'CHROM']) + '_' + str(data_df.loc[i, 'POS'])
        label_dict[i] = label_key_dict[key_]

    label_df = pd.DataFrame.from_dict(label_dict, orient='index', columns=['label'])
    data_df['label'] = label_df['label']
    data_df = data_df.loc[(data_df['label'] != 'NONE') & (data_df['label'] != 'MISS')]
    data_df = data_df.reset_index(drop=True)
    print(data_df)
    return data_df


# make learning data based on preprocessing labeling data
def make_learning_data(data_csv, column_name, learn_data_path):
    feature_column = []
    feature_column_dict = {}
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

        # LABEL
        feature_column_dict[i]['label'] = data_csv.loc[i, 'label']

    input_csv = pd.DataFrame.from_dict(feature_column_dict, orient='index')
    input_csv = input_csv.reset_index(drop=True)
    input_csv.to_csv(learn_data_path)
    return input_csv


if __name__ == '__main__':
    data_list = ['HG001', 'HG002']
    type_list = ['hm.indel', 'hm.snp', 'ht.indel', 'ht.snp']

    for target_data in data_list:
        print(target_data)
        for target_type in type_list:
            print('---------------------------------------------------------')
            print(target_type)

            # 1. first set parameters
            label_path, data_path, learn_data_path, column_name = get_parameters(vcf_dir, data_dir, target_data, target_type)

            # 2. labeling data
            labeled_df = labeling_vcf(label_path, data_path)

            # 3. convert to input csv data
            print('3. convert to input csv data')
            input_csv = make_learning_data(labeled_df, column_name, learn_data_path)
