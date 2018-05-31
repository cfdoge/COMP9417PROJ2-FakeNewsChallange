import numpy as np
import pandas as pd
import pickle
import sys
import sklearn
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from scorer import load_dataset,score_submission,score_defaults,print_confusion_matrix,SCORE_REPORT,FNCException

def test(model):
    df_stance_test = pd.read_csv('competition_test_stances.csv')
    x_test = pd.DataFrame(df_stance_test['Body ID'])
    y_test = df_stance_test.Stance

    # load tf_idf similarity
    with open('test_tfidf_sim', 'rb') as infile:
        tfidf_sim_test = pickle.load(infile)
    x_test['tfidf_sim'] = tfidf_sim_test

    # load svd similarity
    with open('test_svd_sim', 'rb') as infile2:
        svd_sim_test = pickle.load(infile2)
    x_test['svd_sim'] = svd_sim_test

    # load word2vec similarity
    with open('test_w2v_sim', 'rb') as infile3:
        w2v_sim_test = pickle.load(infile3)
    x_test['w2v_sim'] = w2v_sim_test

    # load sentiment feature for head
    with open('test_sentiment_head', 'rb') as infile4:
        sentiment_head_test = pickle.load(infile4)
    x_test['sentiment_head'] = sentiment_head_test

    # load sentiment feature for body
    with open('test_sentiment_body', 'rb') as infile5:
        sentiment_body_test = pickle.load(infile5)

    # load reference id
    with open('test_body_id_reference', 'rb') as infile6:
        ref_id_test = pickle.load(infile6)

    sent_body_list_test = sentiment_body_test.tolist()
    new_sent_body_test = []
    for i in range(len(sentiment_head_test.tolist())):
        new_sent_body_test.append(sent_body_list_test[ref_id_test[i]])

    x_test['sentiment_body'] = pd.Series(new_sent_body_test)

    if x_test.isnull().values.any():
        #x_test[x_test['w2v_sim'].isnull().values]
        x_test.fillna(0.0 ,inplace = True)

        x_test = x_test.drop('Body ID', axis=1)
    y_predict = model.predict(x_test)
    score = sklearn.metrics.accuracy_score(y_test, y_predict)
    print('Accuracy: ', score)

    return score, y_predict

def feedback(y_predict):
    df_comp = pd.read_csv('competition_test_stances_unlabeled.csv', encoding='utf-8')
    df_comp['Stance'] = pd.Series(y_predict)
    df_comp.to_csv('submission.csv', encoding='utf-8',index=False)
    print('Predict values have been saved in the "submission.csv" file')

    try:
        gold_labels = load_dataset('competition_test_stances.csv')
        test_labels = load_dataset('submission.csv')

        test_score, cm = score_submission(gold_labels, test_labels)
        null_score, max_score = score_defaults(gold_labels)
        print_confusion_matrix(cm)
        print(SCORE_REPORT.format(max_score, null_score, test_score))

    except FNCException as e:
        print(e)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('please input the arguments')
        sys.exit(0)

    _, model_name = sys.argv
    # Loading training set
    df_stance = pd.read_csv('train_stances.csv')
    x_train = pd.DataFrame(df_stance['Body ID'])

    # load tf_idf similarity
    with open('tfidf_sim', 'rb') as infile:
        tfidf_sim = pickle.load(infile)
    x_train['tfidf_sim'] = tfidf_sim

    # load svd similarity
    with open('svd_cosine_similarity', 'rb') as infile2:
        svd_sim = pickle.load(infile2)
    x_train['svd_sim'] = svd_sim

    # load word2vec similarity
    with open('w2v_sim', 'rb') as infile3:
        w2v_sim = pickle.load(infile3)
    x_train['w2v_sim'] = w2v_sim

    # load sentiment feature for head
    with open('sentiment_head', 'rb') as infile4:
        sentiment_head = pickle.load(infile4)
    x_train['sentiment_head'] = sentiment_head

    # load sentiment feature for body
    with open('sentiment_body', 'rb') as infile5:
        sentiment_body = pickle.load(infile5)

    # load reference id
    with open('body_id_reference', 'rb') as infile6:
        ref_id = pickle.load(infile6)

    sent_body_list = sentiment_body.tolist()
    new_sent_body = []
    for i in range(len(sentiment_head.tolist())):
        new_sent_body.append(sent_body_list[ref_id[i]])
    x_train['sentiment_body'] = pd.Series(new_sent_body)

    # training prepare
    x_train = x_train.drop('Body ID', axis=1)
    y_train = df_stance.Stance

    if model_name == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        print('Program is fitting and trainig now, please be patient...')
        rfr = RandomForestClassifier(n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features=0.3,
                                     oob_score=True, n_jobs=-1, random_state=3)
        rfr.fit(x_train, y_train)
        print('WOW,such a good model. Start predicting now...')
        # test the model on test set
        acc, y_predict = test(rfr)

        # Got the official feedback
        feedback(y_predict)

    if model_name == 'XGB':
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
        max_depth = range(3, 10, 2)
        min_child_weight = range(1, 6, 2)
        param_test = dict(max_depth=max_depth, min_child_weight=min_child_weight)

        # xgboost parameters tune
        print('Xgb classifier is tuning its parameters, please wait...')
        xgb = XGBClassifier(
            learning_rate=0.1,
            n_estimators=800,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.3,
            colsample_bytree=0.8,
            colsample_bylevel=0.7,
            objective='multi:softprob',
            seed=3)

        gsearch2_1 = GridSearchCV(xgb, param_grid=param_test, scoring='accuracy', n_jobs=-1, cv=kfold)
        gsearch2_1.fit(x_train, y_train)

        print_confusion_matrix('Tuning is finished. Start predicting')
        param_best = gsearch2_1.best_params_
        xgb2 = XGBClassifier(
            learning_rate=0.1,
            n_estimators=800,
            max_depth=param_best['max_depth'],
            min_child_weight=param_best['min_child_weight'],
            gamma=0,
            subsample=0.3,
            colsample_bytree=0.8,
            colsample_bylevel=0.7,
            objective='multi:softprob',
            seed=3)

        xgb2.fit(x_train, y_train)
        # test the model on test set
        acc, y_predict = test(xgb2)
        # Got the official feedback
        feedback(y_predict)

    if model_name == 'LR':
        from sklearn.linear_model import LogisticRegression
        #from sklearn.cross_validation import cross_val_score

        penaltys = ['l1', 'l2']
        Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        tuned_parameters = dict(penalty=penaltys, C=Cs)

        print('Model is running, please be patient...')
        lr_penalty = LogisticRegression()
        gsearch2_1 = GridSearchCV(lr_penalty, tuned_parameters, cv=5, scoring='neg_log_loss')
        gsearch2_1.fit(x_train, y_train)

        p = gsearch2_1.best_params_
        lr = LogisticRegression(penalty=p['penalty'], C = p['C'])
        lr.fit(x_train, y_train)
        # test the model on test set
        acc, y_predict = test(lr)
        # Got the official feedback
        feedback(y_predict)












