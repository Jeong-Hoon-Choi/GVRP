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

            print('2. model inference ROC')
            pred_dict = {}
            for i, model_name in enumerate(models):
                print('2.' + str(i + 1) + ' | model:', model_name)
                if model_name == 'MLP':
                    file_name = model_dir + m + '/' + data_type.replace('.csv', '') + '_' + model_name + '_model.h5'
                    model = tf.keras.models.load_model(file_name)
                    predict_pro_ = model.predict(test_X)
                elif model_name == 'ft_transformer':
                    file_name = model_dir + m + '/' + data_type.replace('.csv', '') + '_' + model_name + '_model.pth'
                    predict_pro_a, predict_ = ft_transformer_inf(file_name, test_X, test_y)
                    predict_pro_ = [pre[1] for pre in predict_pro_a]
                else:
                    file_name = model_dir + m + '/' + data_type.replace('.csv', '') + '_' + model_name + '_model.pkl'
                    model = pickle.load(open(file_name, 'rb'))
                    predict_pro_ = model.predict_proba(test_X)[:, 1]

                m_dict_[model_name] += list(predict_pro_)

                fpr, tpr, threshold_ = roc_curve(test_y, predict_pro_)
                auc_ = auc(fpr, tpr)
                pred_dict[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc_': auc_}

            plt.figure(figsize=(8, 8))
            for m_ in pred_dict:
                plt.plot(pred_dict[m_]['fpr'], pred_dict[m_]['tpr'], lw=2,
                         label=m_ + ' (AUC = %0.3f)' % pred_dict[m_]['auc_'])
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=16)
            plt.ylabel('True Positive Rate', fontsize=16)
            if data_type.split('.')[1] == 'indel':
                ss = 'INDEL'
            else:
                ss = 'SNPs'
            plt.title('ROC curve for ' + m.replace('_', ' ') +
                      '\nvariant type : ' + data_type.split('.')[0].upper() + '-' + ss, fontsize=18)
            plt.legend(loc="lower right", fontsize=12)
            plt.savefig(roc_dir + m + '_' + data_type.replace('.csv', '') + '_roc_curve.png', dpi=600)
