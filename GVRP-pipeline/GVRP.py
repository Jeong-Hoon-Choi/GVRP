from data_handling.before_apply_model import *
from data_handling.after_apply_model import *
from handling_vcf.vcf_handler import *
from learning.model_handling import *


if __name__ == '__main__':
    # 0. setting
    model_dir = './model/'
    parser = argparse.ArgumentParser(description='input output directory')

    # 1. first set parameters
    parser.add_argument('--input', '-i', type=str, required=True, help='input vcf file path')
    parser.add_argument('--output', '-o', type=str, required=True, help='refined vcf file path')
    parser.add_argument('--delete', '-d', type=str, required=False, help='deleted vcf file path')
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    delete_path = args.delete
    print('\ninput vcf path :', input_path)
    print('output refined vcf path :', output_path)
    if delete_path:
        print('output refined vcf path :', output_path)

    # 2. convert vcf to csv
    print('\nread vcf')
    vcf_df = read_vcf(input_path)
    column_name = vcf_df.columns[-1]

    print('\nlabeling variant type')
    data_csv = labeling_df(vcf_df, column_name)

    # 3. convert to feature dataframe
    print('\nextract feature data\n')
    input_csv = make_learning_data(data_csv, column_name)
    # input_csv = pd.read_csv('./new/test/learning/aaa1.csv', index_col=0)

    # 3. apply trained model
    print('\nmodel inference step')
    apply_model(input_csv, data_csv, model_dir)

    print('\ninference result / "0" refers filtering and "1" refers remaining')
    print(data_csv['result'].value_counts())

    # 4. make remaining/filtering set
    remaining_set = get_remaining_filtering_set(data_csv)

    # 5. return to vcf
    print('\nsaving result vcf')
    selecting_columns_for_vcf(input_path, output_path, remaining_set, 'refine')
    if delete_path:
        selecting_columns_for_vcf(input_path, delete_path, remaining_set, 'delete')

    # 6. Finish
    print('\n Done.\n')
