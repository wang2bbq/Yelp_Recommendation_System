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

def item_based_predict(user_id, bus, N):
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
            return rating_predicted

if __name__ == '__main__':
    sc = SparkContext('local[*]', 'item_based')

    start_time = time.time()
    rdd1 = sc.textFile(sys.argv[1]) #'./yelp_train.csv'
    header = rdd1.first()
    trainRDD = rdd1.filter(lambda r: r != header).map(lambda r: r.split(','))

    user_bus_RDD = trainRDD.map(lambda r: (r[0],r[1])).groupByKey().mapValues(set)
    user_bus_dic = {}
    for k,v in user_bus_RDD.collect():
        user_bus_dic[k] = v

    bus_user_RDD = trainRDD.map(lambda r: (r[1], r[0])).groupByKey().mapValues(set)
    bus_user_dic = {}
    for k, v in bus_user_RDD.collect():
        bus_user_dic[k] = v

    user_rating_RDD = (trainRDD.map(lambda r: (r[0],float(r[2])))
                       .groupByKey()
                       .mapValues(list)
                       .mapValues(lambda v: sum(v)/len(v)))
    user_avg_rating_dic = {}
    for k,v in user_rating_RDD.collect():
        user_avg_rating_dic[k] = v
    # get all users
    all_users = {k for k in user_avg_rating_dic.keys()}

    bus_user_ratingRDD = (trainRDD.map(lambda r: (r[1],(r[0],float(r[2]))))
                          .groupByKey()
                          .mapValues(set))
    bus_user_rating_dic = {}
    for k,v in bus_user_ratingRDD.collect():
        dic0 = {}
        for k0,v0 in v:
            dic0[k0] = v0
        bus_user_rating_dic[k] = dic0
    # get all businesses
    all_bus = {k for k in bus_user_rating_dic.keys()}

    # test
    pearson_dict = {}
    rdd2 = sc.textFile(sys.argv[2])  # './yelp_val_in.csv'
    header2 = rdd2.first()
    valRDD = rdd2.filter(lambda r: r != header2).map(lambda r: r.split(','))

    N = 30
    predictRDD = (valRDD
                  .map(lambda r: (r[0],r[1]))
                  .map(lambda t: (t[0], t[1], item_based_predict(t[0], t[1], N))))
    result = predictRDD.collect()

    result_str = 'user_id, business_id, prediction\n'
    for t in result:
        result_str += f'{t[0]},{t[1]},{t[2]}\n'
    with open(sys.argv[3], 'w') as outhand: # './output2.csv'
        outhand.write(result_str)
    end_time = time.time()
    duration = end_time - start_time
    print('Duration:', duration)

# rmse: 1.0487092459212952
# Duration: 51.187041997909546










