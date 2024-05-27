from GVRP_lib import *
warnings.filterwarnings("ignore")


def get_chr_pos_set(target_csv):
    return_list = list()
    for i in range(len(target_csv.index)):
        return_list.append((str(target_csv.loc[i, 'CHROM']), str(target_csv.loc[i, 'POS'])))
    return set(return_list)


def get_remaining_filtering_set(m_df):
    final_csv = m_df.loc[m_df['result'] == 1]
    final_csv = final_csv.reset_index(drop=True)
    final_csv = final_csv.drop(['TYPE', 'result'], axis=1)
    remaining_set = get_chr_pos_set(final_csv)
    return remaining_set


def selecting_columns_for_vcf(original_path, output_path, selection_set, mode):
    print(output_path)
    vcf_reader = cyvcf2.VCF(original_path)
    with open(output_path, 'w') as output_file:
        output_file.write(str(vcf_reader.raw_header))
        for record in vcf_reader:
            if mode == 'refine':
                if (str(record.CHROM), str(record.POS)) in selection_set:
                    output_file.write(str(record))
            else:
                if (str(record.CHROM), str(record.POS)) not in selection_set:
                    output_file.write(str(record))
