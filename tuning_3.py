from pyspark import SparkContext
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import optuna


if __name__ == '__main__':
    sc = SparkContext('local[*]', 'tuning_3')
    sc.setLogLevel('ERROR')

    embed_feature_file1 = './embed_features1.csv'
    embed_feature_file2 = './embed_features2.csv'
    yelp_train_file = './yelp_train.csv'  #
    yelp_val_file = './yelp_val.csv'  #

    rdd11 = sc.textFile(embed_feature_file1)
    header11 = rdd11.first()
    embed_features1 = rdd11.filter(lambda r: r!=header11).map(lambda r: r.split(',')).map(lambda r: [float(i) for i in r]).collect()
    rdd12 = sc.textFile(embed_feature_file2)
    header12 = rdd12.first()
    embed_features2 = rdd12.filter(lambda r: r!=header12).map(lambda r: r.split(',')).map(lambda r: [float(i) for i in r]).collect()
    embed_features = embed_features1 + embed_features2

    rdd2 = sc.textFile(yelp_train_file)
    header2 = rdd2.first()
    trainRDD = rdd2.filter(lambda r: r != header2).map(lambda r: r.split(',')).map(lambda r: [r[0], r[1], float(r[2])])
    y_train = trainRDD.map(lambda r: r[2]).collect()

    rdd3 = sc.textFile(yelp_val_file)  # sys.argv[2]
    header3 = rdd3.first()
    valRDD = rdd3.filter(lambda r: r != header3).map(lambda r: r.split(',')).map(lambda r: [r[0], r[1], float(r[2])])
    y_val = valRDD.map(lambda r: r[2]).collect()

    y_train_val = y_train + y_val
    y_train_val = np.array(y_train_val, dtype=np.float64)
    x_train_val = np.array(embed_features, dtype=np.float64)
    # dtrain = xgb.DMatrix(x_train_val, label=y_train_val)  # DMatrix !!

    x_train = x_train_val[:455854]
    x_train = np.array(x_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    dtrain = xgb.DMatrix(x_train, label=y_train)  # DMatrix

    x_val = x_train_val[455854:]
    x_val = np.array(x_val, dtype=np.float64)
    y_val = np.array(y_val, dtype=np.float64)
    #dval = xgb.DMatrix(x_val)
    dval = xgb.DMatrix(x_val, label=y_val)  # DMatrix

    def xgb_objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.5),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),  #
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-1, 10.0, log=True),
            'objective': 'reg:linear',  # reg:squarederror
            'booster': 'gbtree',
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'max_bin': 256,
            'silent': 1
        }
        evallist = [(dval, 'eval')]
        xgb_model = xgb.train(params, dtrain=dtrain,evals=evallist, early_stopping_rounds=100,verbose_eval=100)

        ypred = xgb_model.predict(dval)
        rmse = mean_squared_error(y_val, ypred, squared=False)
        return rmse

    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(xgb_objective, n_trials=100)

    print('Number of finished trials:', len(study_xgb.trials))
    print('Best trial:', study_xgb.best_trial.params)






