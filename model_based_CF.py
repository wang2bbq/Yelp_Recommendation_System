"""
Method Description:
Use XGBregressor with advanced feature engineering(AutoEncoder) to predict ratings for each user and business pair.
1. Pre-processing: train AutoEncoder model to learn continuous features from high-dimensional "categories" data.
For assignment3 task2-2(last version), I included features like user's "review_count","average_stars","useful","funny","cool" and "fans", business's "stars","review_count" and "is_open".
For this project(model_based), I've added "categories" for businesses, which is a string including multiple categories.
After preprocessing and getting all the possible categories(1305), each category is transformed into a binary feature (0 or 1). Then I trained an AutoEncoder with a 16-neuron hidden layer to derive 16 significant new features. These embedded features are then incorporated into both the training and validation datasets.
2. Hyperparameter tuning: use Optuna to tune hyperparameters of xgboost(using training set).
3. Testing: combine the training set and validation set to train the final model and test on test set

Error Distribution:
>=0 and <1: 103881
>=1 and <2: 31901
>=2 and <3: 5642
>=3 and <4: 620
>=4: 0

RMSE:
1.Train on Training set, test on Validation set -> RMSE: 0.979552469810676
2.Train on Training set & Validation set, test on Validation set -> RMSE: 0.9530645812944019

Execution Time:
99.91s
"""
from pyspark import SparkContext
import numpy as np
import xgboost as xgb
import sys
import time
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    sc = SparkContext('local[*]', 'model_based')
    sc.setLogLevel('ERROR')

    start_time = time.time()
    folder = sys.argv[1]
    yelp_train_file = folder + '/yelp_train.csv'
    yelp_val_file = folder + '/yelp_val.csv'
    yelp_test_file = sys.argv[2]  ###
    output_file = sys.argv[3]

    embed_feature_file1 = './embed_features1.csv'
    embed_feature_file2 = './embed_features2.csv'

    rdd11 = sc.textFile(embed_feature_file1)
    header11 = rdd11.first()
    embed_features1 = rdd11.filter(lambda r: r!=header11).map(lambda r: r.split(',')).map(lambda r: [float(i) for i in r]).collect()
    rdd12 = sc.textFile(embed_feature_file2)
    header12 = rdd12.first()
    embed_features2 = rdd12.filter(lambda r: r!=header12).map(lambda r: r.split(',')).map(lambda r: [float(i) for i in r]).collect()
    embed_features = embed_features1 + embed_features2

    rdd2 = sc.textFile(yelp_train_file)
    header2 = rdd2.first()
    trainRDD = rdd2.filter(lambda r: r!=header2).map(lambda r: r.split(',')).map(lambda r: [r[0],r[1],float(r[2])])
    y_train = trainRDD.map(lambda r: r[2]).collect()

    rdd3 = sc.textFile(yelp_val_file)
    header3 = rdd3.first()
    valRDD = rdd3.filter(lambda r: r != header3).map(lambda r: r.split(',')).map(lambda r: [r[0],r[1],float(r[2])])
    y_val = valRDD.map(lambda r: r[2]).collect()

    # test data
    rdd4 = sc.textFile(yelp_test_file)
    header4 = rdd4.first()
    testRDD = rdd4.filter(lambda r: r != header4).map(lambda r: r.split(',')).map(lambda r: (r[0], r[1]))

    user_id_all = {}
    bus_id_all = {}
    count_ind = 0
    for u,b,r in trainRDD.collect():
        if u not in user_id_all:
            user_id_all[u] = count_ind
        if b not in bus_id_all:
            bus_id_all[b] = count_ind
        count_ind += 1
    for u,b,r in valRDD.collect():
        if u not in user_id_all:
            user_id_all[u] = count_ind
        if b not in bus_id_all:
            bus_id_all[b] = count_ind
        count_ind += 1

    y_train_val = y_train + y_val
    y_train_val = np.array(y_train_val, dtype = np.float64)
    x_train_val = np.array(embed_features, dtype=np.float64)
    dtrain = xgb.DMatrix(x_train_val, label=y_train_val)  # DMatrix !!

    x_train = x_train_val[:455854]
    x_train = np.array(x_train, dtype=np.float64)
    y_train = np.array(y_train, dtype = np.float64)
    #dtrain = xgb.DMatrix(x_train, label=y_train)  # DMatrix

    x_val = x_train_val[455854:]
    x_val = np.array(x_val, dtype=np.float64)
    dval = xgb.DMatrix(x_val)
    #dval = xgb.DMatrix(x_val, label=y_val)  # DMatrix

    params = {
        'learning_rate': 0.06, #0.1 !! 0.08
        'max_depth': 7,
        'subsample': 0.8826063703166193,
        'colsample_bytree': 0.8344140741655501,
        'reg_lambda': 6.875729856312586,  # l2
        'objective': 'reg:linear',  # reg:squarederror
        'booster': 'gbtree',
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'max_bin': 256,
        'silent': 1
    }
    #evallist = [(dval, 'eval')]
    xgb_model = xgb.train(params,
              dtrain=dtrain,
              num_boost_round=383)
              #evals=evallist,early_stopping_rounds=100,verbose_eval=100)  # 166 (161 is best) 216

    # generate test data(features)
    ypred = []
    for u,b in testRDD.collect():
        if u in user_id_all and b in bus_id_all:
            user_feature = embed_features[user_id_all[u]][:6]
            bus_feature = embed_features[bus_id_all[b]][6:]
            row_feature = user_feature + bus_feature
            x_single = xgb.DMatrix([row_feature])
            rating = xgb_model.predict(x_single)
            rating = rating[0]
        elif u in user_id_all and b not in bus_id_all:
            user_feature = embed_features[user_id_all[u]][:6]
            rating = user_feature[1]
        elif u not in user_id_all and b in bus_id_all:
            bus_feature = embed_features[bus_id_all[b]][6:]
            rating = bus_feature[0]
        else:
            rating = 3.75
        ypred.append(rating)

    result_str = 'user_id, business_id, prediction\n'
    test_pair = testRDD.collect()
    for i in range(len(ypred)):
        user_id = test_pair[i][0]
        bus_id = test_pair[i][1]
        rating_pred = ypred[i]
        result_str += f'{user_id},{bus_id},{rating_pred}\n'

    with open(output_file, 'w') as outhand: # './competition_output.csv'
        outhand.write(result_str)

    end_time = time.time()
    duration = end_time - start_time
    print('Duration:', duration)

    #rmse = mean_squared_error(y_val, ypred, squared=False)
    #print("RMSE: ", rmse)

    #count_01 = 0
    #count_12 = 0
    #count_23 = 0
    #count_34 = 0
    #count_4 = 0
    #for i in range(len(ypred)):
        #error_abs = abs(ypred[i]-y_val[i])
        #if error_abs>=0 and error_abs<1:
            #count_01+=1
        #elif error_abs>=1 and error_abs<2:
            #count_12+=1
        #elif error_abs>=2 and error_abs<3:
            #count_23+=1
        #elif error_abs>=3 and error_abs<4:
            #count_34+=1
        #elif error_abs>=4:
            #count_4+=1
    #print('count_01',count_01)
    #print('count_12',count_12)
    #print('count_23',count_23)
    #print('count_34',count_34)
    #print('count_4',count_4)






