from common_module import *


if __name__ == '__main__':
    for m in mode:
        print('-----------------------------------------------------------------------------')
        print('mode :', m)

        result_dict_ = {'f1': [], 'precision': [], 'recall': [], 'Accuracy':[]}
        pred_dict_ = {}
        models = ['XG_Boost', 'LGBM', 'RF', 'LR', 'KNN', 'NB', 'MLP', 'ft_transformer']
        m_dict_ = {}
        l_dict_ = {'label': []}
        for mo in models:
            m_dict_[mo] = list()

        for data_type in data_type_list:
            print('\ndata type :', data_type)
            print('1. load train test data')
            train_X, train_y, test_X, test_y = load_data(m, data_dir, data_type)

            l_dict_['label'] += list(test_y)

            result_dict = {'f1': [], 'precision': [], 'recall': [], 'Accuracy':[]}
            print('2. model learning')
            for i, model_name in enumerate(models):
                print('2.' + str(i + 1) + ' | model:', model_name)
                model, pred = eval(model_name)(train_X, test_X, train_y, test_y)
                score_f(test_y, pred, result_dict)
                if model_name == 'MLP':
                    file_name = model_dir + m + '/' + data_type.replace('.csv', '') + '_' + model_name + '_model.h5'
                    model.save(file_name)
                elif model_name == 'ft_transformer':
                    file_name = model_dir + m + '/' + data_type.replace('.csv', '') + '_' + model_name + '_model.pth'
                    torch.save(model.state_dict(), file_name)
                else:
                    file_name = model_dir + m + '/' + data_type.replace('.csv', '') + '_' + model_name + '_model.pkl'
                    pickle.dump(model, open(file_name, 'wb'))
            label_df = pd.DataFrame.from_dict(result_dict, orient='index', columns=models)
            label_df = label_df.transpose()
            print()
            print('type :', data_type)
            print('#train :', len(train_y), '#test :', len(test_y), '\n')
            print(tabulate(label_df, headers='keys', tablefmt='psql'))
            label_df.to_csv(performance_dir + m + '_' + data_type + '_result.csv')
