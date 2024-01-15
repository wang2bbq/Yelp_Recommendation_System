from pyspark import SparkContext
import math
import sys
import time

def pearson_correlation(ratings1, ratings2, mean1_all, mean2_all): #all ratings
    up = 0
    sum_sq1 = 0
    sum_sq2 = 0
    for i in range(len(ratings1)):
        rating1_avg = ratings1[i] - mean1_all
        rating2_avg = ratings2[i] - mean2_all
        up += rating1_avg * rating2_avg
        sum_sq1 += rating1_avg**2
        sum_sq2 += rating2_avg**2
    sqrt1 = math.sqrt(sum_sq1)
    sqrt2 = math.sqrt(sum_sq2)
    if sqrt1 == 0 or sqrt2 == 0:
        w0_list = []
        for i in range(len(ratings1)):
            w0 = 1-abs(ratings1[i]-ratings2[i])/5
            w0_list.append(w0)
        pearson = sum(w0_list)/len(w0_list)
    else:
        pearson = up / (sqrt1 * sqrt2)
    return pearson

def item_based_predict(user_id, bus, N): # version2
    global pearson_dict
    # cold start: new user and new item(business)
    if user_id not in all_users and bus not in all_bus:
        return 3.75
    # cold start: new user
    elif user_id not in all_users:
        return 3.75
    # cold start: new item(business): use user's avg rating
    elif bus not in all_bus:
        rating_predicted = user_avg_rating_dic[user_id]
        return rating_predicted
    else: # user_id in all_users and bus_id in all_bus:
        weight_list = []
        for b0 in user_bus_dic[user_id]:
            pair = tuple(sorted((bus, b0)))
            if pair in pearson_dict.keys():
                weight = pearson_dict[pair]
            else:
                co_users = bus_user_dic[bus] & bus_user_dic[b0]
                if len(co_users) == 0:
                    all_rating1 = list(bus_user_rating_dic[bus].values())
                    all_rating2 = list(bus_user_rating_dic[b0].values())
                    all_mean1 = sum(all_rating1) / len(all_rating1)
                    all_mean2 = sum(all_rating2) / len(all_rating2)
                    weight = 1 - abs(all_mean1 - all_mean2) / 5
                    if weight <= 0.8:  # tune!!! 0.8->0.9->0.7->0.8
                        continue
                    weight = weight * (abs(weight) ** 1.5)  # Case Amplification
                elif len(co_users) == 1:
                    co_u = list(co_users)[0]
                    co_rating1 = bus_user_rating_dic[bus][co_u]
                    co_rating2 = bus_user_rating_dic[b0][co_u]
                    weight = 1 - abs(co_rating1 - co_rating2) / 5
                    if weight <= 0.6:  # tune!!! 0.5->0.6
                        continue
                    weight = weight * (abs(weight) ** 1.5)  # Case Amplification
                elif len(co_users) == 2:
                    b1_coratings = []
                    b2_coratings = []
                    for u in co_users:
                        b1_coratings.append(bus_user_rating_dic[bus][u])
                        b2_coratings.append(bus_user_rating_dic[b0][u])
                    w1 = 1 - abs(b1_coratings[0] - b2_coratings[0]) / 5
                    w2 = 1 - abs(b1_coratings[1] - b2_coratings[1]) / 5
                    weight = (w1 + w2) / 2
                    if weight <= 0.7:  # tune!!! 0.5->0.6->0.7
                        continue
                    weight = weight * (abs(weight) ** 1.5)  # Case Amplification
                else:
                    b1_allratings = list(bus_user_rating_dic[bus].values())
                    b2_allratings = list(bus_user_rating_dic[b0].values())
                    mean1_all = sum(b1_allratings) / len(b1_allratings)
                    mean2_all = sum(b2_allratings) / len(b2_allratings)
                    b1_coratings = []
                    b2_coratings = []
                    for u in co_users:
                        b1_coratings.append(bus_user_rating_dic[bus][u])
                        b2_coratings.append(bus_user_rating_dic[b0][u])
                    weight = pearson_correlation(b1_coratings, b2_coratings, mean1_all, mean2_all)
                    if weight < 0.1: # 0.1->0.2->0.05->0.01->0.1
                        continue
                    weight = weight * (abs(weight) ** 1.5)  # Case Amplification
                pearson_dict[pair] = weight
            weight_list.append((weight, bus_user_rating_dic[b0][user_id]))
        sorted_weight_list = sorted(weight_list, key=lambda t:t[0], reverse=True)
        up = 0
        sum_abw = 0
        neighbor_count = 0
        for w, r in sorted_weight_list:
            up += (w*r)
            sum_abw += abs(w)
            neighbor_count += 1
            if neighbor_count == N:  # N: number of neighbors we want to use
                break
        if sum_abw == 0: # => new item
            rating_predicted = user_avg_rating_dic[user_id]
            return rating_predicted
        else:
            rating_predicted = up / sum_abw
            return rating_predicted, neighbor_count

if __name__ == '__main__':
    sc = SparkContext('local[*]', 'model_based')
    sc.setLogLevel('ERROR')

    start_time = time.time()

    # 1.model-based CF
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

    # 2.item-based CF
    user_bus_RDD = trainRDD.map(lambda r: (r[0], r[1])).groupByKey().mapValues(set)
    user_bus_dic = {}
    for k, v in user_bus_RDD.collect():
        user_bus_dic[k] = v
    bus_user_RDD = trainRDD.map(lambda r: (r[1], r[0])).groupByKey().mapValues(set)
    bus_user_dic = {}
    for k, v in bus_user_RDD.collect():
        bus_user_dic[k] = v
    user_rating_RDD = (trainRDD.map(lambda r: (r[0], float(r[2])))
                       .groupByKey()
                       .mapValues(list)
                       .mapValues(lambda v: sum(v) / len(v)))
    user_avg_rating_dic = {}
    for k, v in user_rating_RDD.collect():
        user_avg_rating_dic[k] = v
    # get all users
    all_users = {k for k in user_avg_rating_dic.keys()}

    bus_user_ratingRDD = (trainRDD.map(lambda r: (r[1], (r[0], float(r[2]))))
                          .groupByKey()
                          .mapValues(set))
    bus_user_rating_dic = {}
    for k, v in bus_user_ratingRDD.collect():
        dic0 = {}
        for k0, v0 in v:
            dic0[k0] = v0
        bus_user_rating_dic[k] = dic0
    # get all businesses
    all_bus = {k for k in bus_user_rating_dic.keys()}
    # test
    pearson_dict = {}
    N = 30

    # 3. hybrid
    ypred = []
    for u,b in testRDD.collect():
        if u in user_id_all and b in bus_id_all:
            user_feature = embed_features[user_id_all[u]][:6]
            bus_feature = embed_features[bus_id_all[b]][6:]
            row_feature = user_feature + bus_feature
            # item_based predict
            rating_i, num_neighbor = item_based_predict(u, b, N)
            # model_based predict
            x_single = xgb.DMatrix([row_feature])
            rating_m = xgb_model.predict(x_single)
            rating_m = rating_m[0]
            # check and hybrid
            if num_neighbor < 30:
                alpha = 0.05
            else:
                alpha = 0.15
            rating = alpha * rating_i + (1-alpha) * rating_m
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

    with open(output_file, 'w') as outhand:  # './competition_output.csv'
        outhand.write(result_str)

    end_time = time.time()
    duration = end_time - start_time
    print('Duration:', duration)

