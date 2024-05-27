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

            print('2. model inference performance')
            for i, model_name in enumerate(models):
                print('2.' + str(i + 1) + ' | model:', model_name)
                if model_name == 'MLP':
                    file_name = model_dir + m + '/' + data_type.replace('.csv', '') + '_' + model_name + '_model.h5'
                    model = tf.keras.models.load_model(file_name)
                    predict_ = model.predict(test_X)
                    predict_ = np.where(predict_ > 0.5, 1, 0)
                elif model_name == 'ft_transformer':
                    file_name = model_dir + m + '/' + data_type.replace('.csv', '') + '_' + model_name + '_model.pth'
                    predict_pro_, predict_ = ft_transformer_inf(file_name, test_X, test_y)
                else:
                    file_name = model_dir + m + '/' + data_type.replace('.csv', '') + '_' + model_name + '_model.pkl'
                    model = pickle.load(open(file_name, 'rb'))
                    predict_ = model.predict(test_X)

                m_dict_[model_name] += list(predict_)

        for mo_ in m_dict_:
            score_f(l_dict_['label'], m_dict_[mo_], result_dict_)

        label_df = pd.DataFrame.from_dict(result_dict_, orient='index', columns=models)
        label_df = label_df.transpose()
        print()
        print('type :', 'total_snps')
        print(tabulate(label_df, headers='keys', tablefmt='psql'))
        label_df.to_csv(performance_dir + m + '_total_result.csv')
