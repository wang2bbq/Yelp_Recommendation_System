from pyspark import SparkContext
import numpy as np
import csv
import json

if __name__ == '__main__':
    sc = SparkContext('local[*]', 'preprocessing_1')
    sc.setLogLevel('ERROR')

    business_file = './business.json'
    user_file = './user.json'
    yelp_train_file = './yelp_train.csv'
    rdd0 = (sc.textFile(business_file)
            .map(lambda r: json.loads(r))
            .filter(lambda r: r['categories'])
            .flatMap(lambda r:r['categories'].split(', ')))
    cates = set(rdd0.collect())
    catel = list(cates)

    rdd1 = (sc.textFile(business_file)
            .map(lambda r: json.loads(r))
            .map(lambda r: (r['business_id'],(r['stars'],r['review_count'],r['is_open'],r['categories']))))

    rdd2 = (sc.textFile(user_file).map(lambda r: json.loads(r))
            .map(lambda r: (r['user_id'],(r['review_count'],r['average_stars'],r['useful'],r['funny'],r['cool'],r['fans']))))
    rdd3 = sc.textFile(yelp_train_file)
    header = rdd3.first()
    trainRDD = rdd3.filter(lambda r: r!=header).map(lambda r: r.split(',')).map(lambda r: [r[0],r[1],float(r[2])])

    bus_dict = {}
    for k,v in rdd1.collect():
        bus_dict[k] = v
    user_dict = {}
    for k,v in rdd2.collect():
        user_dict[k] = v

    x_train = []
    y_train = []

    for u,b,r in trainRDD.collect():
        y_train.append(r)
        if u not in user_dict.keys():
            review_count_u = None
            average_stars = None
            useful = None
            funny = None
            cool = None
            fans = None
        else:
            review_count_u = user_dict[u][0]
            average_stars = user_dict[u][1]
            useful = user_dict[u][2]
            funny = user_dict[u][3]
            cool = user_dict[u][4]
            fans = user_dict[u][5]
        if b not in bus_dict.keys():
            stars = None
            review_count_b = None
            is_open = None
            cate_dict = {key: None for key in catel}
        else:
            stars = bus_dict[b][0]
            review_count_b = bus_dict[b][1]
            is_open = bus_dict[b][2]
            cate_dict = {key: 0 for key in catel}
            if bus_dict[b][3]:
                l = bus_dict[b][3].split(', ')
                for c in l:
                    cate_dict[c] = 1
            else:
                cate_dict = {key: None for key in catel}
        x_train.append((review_count_u,average_stars,useful,funny,cool,fans,stars,review_count_b,is_open)+tuple(cate_dict.values()))


    # test
    rdd4 = sc.textFile('./yelp_val.csv') # sys.argv[2]
    header2 = rdd4.first()
    valRDD = rdd4.filter(lambda r: r != header2).map(lambda r: r.split(',')).map(lambda r: (r[0],r[1]))
    val_user_bus = valRDD.collect()
    x_val = []
    for u,b in val_user_bus:
        if u not in user_dict.keys():
            review_count_u = None
            average_stars = None
            useful = None
            funny = None
            cool = None
            fans = None
        else:
            review_count_u = user_dict[u][0]
            average_stars = user_dict[u][1]
            useful = user_dict[u][2]
            funny = user_dict[u][3]
            cool = user_dict[u][4]
            fans = user_dict[u][5]
        if b not in bus_dict.keys():
            stars = None
            review_count_b = None
            is_open = None
            cate_dict = {key: None for key in catel} #
        else:
            stars = bus_dict[b][0]
            review_count_b = bus_dict[b][1]
            is_open = bus_dict[b][2]
            cate_dict = {key: 0 for key in catel}
            if bus_dict[b][3]:
                l = bus_dict[b][3].split(', ')
                for c in l:
                    cate_dict[c] = 1
            else:
                cate_dict = {key: None for key in catel}
        x_val.append((review_count_u,average_stars,useful,funny,cool,fans,stars,review_count_b,is_open)+tuple(cate_dict.values()))

    x_train = np.array(x_train)
    x_val = np.array(x_val)

    with open('./med_train.csv', "w") as csvfile1:
        writer1 = csv.writer(csvfile1)
        for row in x_train:
            writer1.writerow(row)
    with open('./med_val.csv', "w") as csvfile2:
        writer2 = csv.writer(csvfile2)
        for row in x_val:
            writer2.writerow(row)






