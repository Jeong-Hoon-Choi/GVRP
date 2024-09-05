from GVRP_lib import *


def read_vcf(path):
    with open(path, 'r') as f:
        lines_info = [l for l in tqdm(f) if not l.startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines_info)),
        dtype={'#CHROM': str, 'POS': str, 'ID': str, 'REF': str, 'ALT': str, 'QUAL': float, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})
