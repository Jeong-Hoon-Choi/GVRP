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


def extract_alignment_info(bam, chrom, pos):
    cigar_features = []
    mapq_values = []
    strand_counts = {"forward": 0, "reverse": 0}  # Strand count 초기화

    for read in bam.fetch(str(chrom), int(pos) - 1, int(pos)):  # 0-based 좌표
        # Strand 정보 계산
        if read.flag & 16:  # Reverse strand
            strand_counts["reverse"] += 1
        else:  # Forward strand
            strand_counts["forward"] += 1

        # CIGAR 분석
        cigar_string = read.cigarstring
        if cigar_string is None:
            continue

        # Count CIGAR operations
        operations = re.findall(r'(\d+)([MIDNSHP=X])', cigar_string)
        feature = {"M": 0, "S": 0, "I": 0, "D": 0}
        for length, op in operations:
            if op in feature:
                feature[op] += int(length)

        feature["Total_length"] = sum(feature.values())  # 리드 총 길이
        feature["M_ratio"] = feature["M"] / feature["Total_length"] if feature["Total_length"] > 0 else 0
        feature["S_ratio"] = feature["S"] / feature["Total_length"] if feature["Total_length"] > 0 else 0
        feature["CIGAR"] = cigar_string

        # MAPQ
        mapq_values.append(read.mapping_quality)

        # Merge features
        cigar_features.append(feature)

    # 평균 CIGAR feature 계산
    avg_cigar = {
        "M_ratio": np.mean([feat["M_ratio"] for feat in cigar_features]) if cigar_features else None,
        "S_ratio": np.mean([feat["S_ratio"] for feat in cigar_features]) if cigar_features else None,
        "Total_reads": len(cigar_features),
    }

    # MAPQ 통계 계산
    mapq_stats = {
        "mean_mapq": np.mean(mapq_values) if mapq_values else None,
        "low_mapq_ratio": sum(1 for mq in mapq_values if mq < 20) / len(mapq_values) if mapq_values else None,
    }

    # Strand 비율 계산
    total_strands = strand_counts["forward"] + strand_counts["reverse"]
    strand_ratios = {
        "forward_strand_ratio": strand_counts["forward"] / total_strands if total_strands > 0 else 0,
        "reverse_strand_ratio": strand_counts["reverse"] / total_strands if total_strands > 0 else 0,
    }

    return {**avg_cigar, **mapq_stats, **strand_ratios}


def get_alignment_information(seq_dir, labeled_df):
    align_info_results = []
    chroms = labeled_df['CHROM'].values
    positions = labeled_df['POS'].values

    with pysam.AlignmentFile(seq_dir, "rb") as bam:
        for i in tqdm(range(len(labeled_df.index))):
            chrom = chroms[i]
            pos = positions[i]
            align_info = extract_alignment_info(bam, chrom, pos)
            align_info_results.append({"CHROM": chrom, "POS": pos, **align_info})

    # align_info_results를 DataFrame으로 변환
    align_info_df = pd.DataFrame(align_info_results)

    # labeled_df와 병합
    align_df = labeled_df.merge(align_info_df, on=["CHROM", "POS"], how="left")
    align_df.fillna({
        "M_ratio": 0, "S_ratio": 0, "Total_reads": 0,
        "mean_mapq": 0, "low_mapq_ratio": 0,
        "forward_strand_ratio": 0, "reverse_strand_ratio": 0
    }, inplace=True)
    return align_df


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
