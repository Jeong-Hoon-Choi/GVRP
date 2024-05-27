from common_parameter import *


# load data
def load_data(m, data_path, data_type):
    if m == 'train_HG001_test_HG002':
        train_data = pd.read_csv(data_path + 'HG001' + data_type, index_col=0)
        train_X = train_data.drop(drop_features, axis=1)
        train_y = train_data[label]

        test_data = pd.read_csv(data_path + 'HG002' + data_type, index_col=0)
        test_X = test_data.drop(drop_features, axis=1)
        test_y = test_data[label]
    elif m == 'train_HG002_test_HG001':
        train_data = pd.read_csv(data_path + 'HG002' + data_type, index_col=0)
        train_X = train_data.drop(drop_features, axis=1)
        train_y = train_data[label]

        test_data = pd.read_csv(data_path + 'HG001' + data_type, index_col=0)
        test_X = test_data.drop(drop_features, axis=1)
        test_y = test_data[label]
    elif m == 'mixed_data':
        data_1 = pd.read_csv(data_path + 'HG001' + data_type, index_col=0)
        data_2 = pd.read_csv(data_path + 'HG002' + data_type, index_col=0)
        data = pd.concat([data_1, data_2]).reset_index(drop=True)
        data_X = data.drop(drop_features, axis=1)
        data_y = data[label]
        train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.5, random_state=1024, stratify=data_y)
    elif m == 'concat_data_train_HG001_test_HG002':
        train_data = pd.read_csv(data_path + 'HG001_concat_all.csv', index_col=0)
        train_X = train_data.drop(drop_features, axis=1)
        train_y = train_data[label]

        test_data = pd.read_csv(data_path + 'HG002_concat_all.csv', index_col=0)
        test_X = test_data.drop(drop_features, axis=1)
        test_y = test_data[label]
    elif m == 'concat_data_train_HG002_test_HG001':
        train_data = pd.read_csv(data_path + 'HG002_concat_all.csv', index_col=0)
        train_X = train_data.drop(drop_features, axis=1)
        train_y = train_data[label]

        test_data = pd.read_csv(data_path + 'HG001_concat_all.csv', index_col=0)
        test_X = test_data.drop(drop_features, axis=1)
        test_y = test_data[label]
    elif m == 'mix_all':
        data_2 = pd.read_csv(data_path + 'HG001_concat_all.csv', index_col=0)
        data_1 = pd.read_csv(data_path + 'HG002_concat_all.csv', index_col=0)
        data = pd.concat([data_1, data_2]).reset_index(drop=True)
        data_X = data.drop(drop_features, axis=1)
        data_y = data[label]
        train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.5, random_state=1024, stratify=data_y)
    else:
        print('wrong')
        exit('1')
    return train_X, train_y, test_X, test_y


# XGBoost model
def XG_Boost(X_train, X_test, y_train, y_test):
    xgb = XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.05, tree_method='gpu_hist')
    # params = {
    #     "n_estimators": [200, 500, 1000],
    #     "max_depth": [5, 7, 10],
    #     "learning_rate": [0.1, 0.075, 0.05, 0.025, 0.01]
    # }
    # xgb_grid = GridSearchCV(xgb, params, cv=5, n_jobs=20, verbose=1)
    # xgb_grid.fit(X_train, y_train)
    # xgb_best = xgb_grid.best_estimator_
    # print(xgb_best)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_pred_proba = xgb.predict_proba(X_test)
    return xgb, xgb_pred


# LightGBM model
def LGBM(X_train, X_test, y_train, y_test):
    lgbm = LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05)
    # params = {
    #     "n_estimators": [200, 500, 1000],
    #     "max_depth": [5, 7, 10],
    #     "learning_rate": [0.1, 0.075, 0.05, 0.025, 0.01]
    # }
    # lgbm_grid = GridSearchCV(lgbm, params, cv=5, n_jobs=20, verbose=1)
    # lgbm_grid.fit(X_train, y_train)
    # lgbm_best = lgbm_grid.best_estimator_
    # print(lgbm_best)
    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_test)
    lgbm_pred_proba = lgbm.predict_proba(X_test)
    return lgbm, lgbm_pred


# Random Forest model
def RF(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=500, max_depth=10)
    # params = {
    #     "n_estimators": [50, 200, 500],
    #     "max_depth": [5, 7, 10]
    # }
    # rf_grid = GridSearchCV(rf, params, cv=5, n_jobs=20, verbose=1)
    # rf_grid.fit(X_train, y_train)
    # rf_best = rf_grid.best_estimator_
    # print(rf_best)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_pred_proba = rf.predict_proba(X_test)
    return rf, rf_pred


def SVM(X_train, X_test, y_train, y_test):
    svm = SVC()
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_pred_proba = svm.predict_proba(X_test)
    return svm, svm_pred


def LR(X_train, X_test, y_train, y_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_pred_proba = lr.predict_proba(X_test)
    return lr, lr_pred


def KNN(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=100)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_pred_proba = knn.predict_proba(X_test)
    return knn, knn_pred


def NB(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_pred_proba = nb.predict_proba(X_test)
    return nb, nb_pred


def MLP(X_train, X_test, y_train, y_test):
    mlp = FFN(X_train.shape[1], X_train, y_train, epochs=800)
    mlp_model = mlp.run()
    mlp_pred = mlp_model.predict(x=X_test)
    mlp_pred = np.where(mlp_pred > 0.5, 1, 0)
    return mlp_model, mlp_pred


def ft_transformer(X_train, X_test, y_train, y_test):
    ftt, fft_pred = FTT(X_train, y_train, X_test, y_test)
    return ftt, fft_pred


def ft_transformer_inf(model_dir, y_train, y_test):
    return FFT_inf(model_dir, y_train, y_test)


def score_f(y_test, pred, result_dict):
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    accuracy = accuracy_score(y_test, pred)

    result_dict['f1'].append(f1)
    result_dict['precision'].append(precision)
    result_dict['recall'].append(recall)
    result_dict['Accuracy'].append(accuracy)
