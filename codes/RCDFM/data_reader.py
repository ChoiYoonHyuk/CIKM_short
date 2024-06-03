import random
import json
from tqdm import tqdm
import numpy as np
from string import punctuation

data_size = 1.0


def read_dataset(s_path, t_path):
    s_dict, t_dict, w_embed = dict(), dict(), dict()
    s_data, t_train, t_valid, t_test = [], [], [], []
    len_t_data = 0

    print('\nLoading source data ... \n')

    f = open(s_path, 'r')

    while True:
        # Read one line & Learn
        line = f.readline()
        if not line: break

        # Convert str to json format
        line = json.loads(line)

        try:
            user = line['reviewerID']
            item = line['asin']
            review = line['reviewText']
            rating = line['overall']

            review = review.lower()
            # Remove punctuation
            review = ''.join([c for c in review if c not in punctuation])

        except KeyError:
            continue

        s_data.append([user, item, rating])

        if user in s_dict:
            s_dict[user].append([item, review])
        else:
            s_dict[user] = [[item, review]]

        if item in s_dict:
            s_dict[item].append([user, review])
        else:
            s_dict[item] = [[user, review]]
    f.close()

    f = open(t_path, 'r')
    while True:
        len_t_data += 1
        # Read one line & Learn
        line = f.readline()
        if not line: break

    len_train_data = int(len_t_data * data_size)
    len_t_data = int(len_t_data * 0.2)
    f.close()
    len_u, len_i = [], []

    f = open(t_path, 'r')
    while True:
        # Read one line & Learn
        line = f.readline()
        if not line: break

        # Convert str to json format
        line = json.loads(line)

        try:
            user = line['reviewerID']
            item = line['asin']
            #review = line['reviewText']
            review = line['generatedSummary']
            rating = line['overall']
            
            len_u.append(user)
            len_i.append(item)

            review = review.lower()
            # Remove punctuation
            review = ''.join([c for c in review if c not in punctuation])

        except KeyError:
            continue

        if user in t_dict and item in t_dict and len(t_valid) < len_t_data:
            t_valid.append([user, item, rating])
        else:
            if len(t_train) > len_train_data:
                break

            t_train.append([user, item, rating])

            if user in t_dict:
                t_dict[user].append([item, review])
            else:
                t_dict[user] = [[item, review]]
            if item in t_dict:
                t_dict[item].append([user, review])
            else:
                t_dict[item] = [[user, review]]

    f.close()

    t_test = t_valid[int(len_t_data/2):len_t_data]
    t_valid = t_valid[0:int(len_t_data/2)]

    print(len(t_train), len(t_valid), len(t_test))
    print(len(list(set(len_u))), len(list(set(len_i))))

    f = open('../meta_learning/data/glove.6B.100d.txt')

    for line in f:
        word_vector = line.split()
        word = word_vector[0]
        word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
        w_embed[word] = word_vector_arr

    f.close()

    return s_data, s_dict, t_train, t_valid, t_test, t_dict, w_embed


def read_baseline_data(t_path):
    # Initialization
    t_dict, w_embed = dict(), dict()
    t_train, t_valid, t_test = [], [], []
    len_t_data = 0
    
    f = open(t_path, 'r')
    while True:
        len_t_data += 1
        line = f.readline()
        if not line: break

    len_train_data = int(len_t_data * 0.8)
    len_t_data = int(len_t_data * 0.2)
    f.close()
    
    # Read target domain's data
    f = open(t_path, 'r')
    while True:
        line = f.readline()
        if not line: break

        line = json.loads(line)

        try:
            user, item, review, rating = line['reviewerID'], line['asin'], line['reviewText'], line['overall']

            review = review.lower()
            review = ''.join([c for c in review if c not in punctuation])

        except KeyError:
            continue

        if user in t_dict and item in t_dict and len(t_valid) < len_t_data:
            t_valid.append([user, item, rating])
        else:
            if len(t_train) > len_train_data:
                break

            t_train.append([user, item, rating])

            if user in t_dict:
                t_dict[user].append([item, review])
            else:
                t_dict[user] = [[item, review]]
            if item in t_dict:
                t_dict[item].append([user, review])
            else:
                t_dict[item] = [[user, review]]

    f.close()

    # Split valid / test data
    t_test, t_valid = t_valid[int(len_t_data/2):len_t_data], t_valid[0:int(len_t_data/2)]

    print('Size of Train / Valid / Test data  : %d / %d / %d' % (len(t_train), len(t_valid), len(t_test)))

    # Dictionary for word embedding
    f = open('../meta_learning/data/glove.6B.100d.txt')

    for line in f:
        word_vector = line.split()
        word = word_vector[0]
        word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
        w_embed[word] = word_vector_arr

    f.close()

    return t_train, t_valid, t_test, t_dict, w_embed


def read_yelp_data(s_path, t_path):
    print('\nLoading data ... \n')

    s_dict, t_dict, w_embed = dict(), dict(), dict()
    s_data, t_train, t_valid, t_test = [], [], [], []
    len_t_data = 0

    f = open(s_path, 'r')

    while True:
        # Read one line & Learn
        line = f.readline()
        if not line: break

        # Convert str to json format
        line = json.loads(line)

        try:
            user = line['user_id']
            item = line['business_id']
            review = line['text']
            rating = line['stars']
            
            review = review.lower()
            # Remove punctuation
            review = ''.join([c for c in review if c not in punctuation])

        except KeyError:
            continue

        s_data.append([user, item, rating])

        if user in s_dict:
            s_dict[user].append([item, review])
        else:
            s_dict[user] = [[item, review]]

        if item in s_dict:
            s_dict[item].append([user, review])
        else:
            s_dict[item] = [[user, review]]
    f.close()

    f = open(t_path, 'r')
    while True:
        len_t_data += 1
        # Read one line & Learn
        line = f.readline()
        if not line: break

    len_train_data = int(len_t_data * data_size)
    len_t_data = int(len_t_data * 0.2)
    f.close()
    
    f = open(t_path, 'r')
    while True:
        # Read one line & Learn
        line = f.readline()
        if not line: break

        # Convert str to json format
        line = json.loads(line)

        try:
            user = line['reviewerID']
            item = line['asin']
            review = line['reviewText']
            rating = line['overall']
            
            review = review.lower()
            # Remove punctuation
            review = ''.join([c for c in review if c not in punctuation])

        except KeyError:
            continue

        if user in t_dict and item in t_dict and len(t_valid) < len_t_data:
            t_valid.append([user, item, rating])
        else:
            '''if len(t_train) > len_train_data:
                break'''

            t_train.append([user, item, rating])

            if user in t_dict:
                t_dict[user].append([item, review])
            else:
                t_dict[user] = [[item, review]]
            if item in t_dict:
                t_dict[item].append([user, review])
            else:
                t_dict[item] = [[user, review]]

    f.close()

    t_test = t_valid[int(len_t_data/2):len_t_data]
    t_valid = t_valid[0:int(len_t_data/2)]

    print(len(t_train), len(t_valid), len(t_test))

    f = open('../meta_learning/data/glove.6B.100d.txt')

    for line in f:
        word_vector = line.split()
        word = word_vector[0]
        word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
        w_embed[word] = word_vector_arr

    f.close()

    return s_data, s_dict, t_train, t_valid, t_test, t_dict, w_embed


def get_rating_matrix(t_path):
    len_t_data = 0
    f = open(t_path, 'r')
    t_user, t_item, t_user_list, t_item_list = dict(), dict(), [], []
    t_train, t_valid = [], []

    while True:
        len_t_data += 1
        # Read one line & Learn
        line = f.readline()
        if not line: break

        # Convert str to json format
        line = json.loads(line)

        t_user_list.append(line['reviewerID'])
        t_item_list.append(line['asin'])

    len_t_data = int(len_t_data * 0.2)
    f.close()

    t_user_list = list(set(t_user_list))
    t_item_list = list(set(t_item_list))

    t_u_dim = len(t_item_list)
    t_i_dim = len(t_user_list)

    t_u_init = [0.0] * t_u_dim
    t_i_init = [0.0] * t_i_dim

    f = open(t_path, 'r')
    while True:
        # Read one line & Learn
        line = f.readline()
        if not line: break

        # Convert str to json format
        line = json.loads(line)

        user = line['reviewerID']
        item = line['asin']
        rating = line['overall']

        if user in t_user and item in t_item and len(t_valid) < len_t_data:
            t_valid.append([user, item, rating])

            if user in t_user:
                idx = t_item_list.index(item)
                t_user[user][idx] = 0.0
            else:
                t_user[user] = t_u_init
                idx = t_item_list.index(item)
                t_user[user][idx] = 0.0
            if item in t_item:
                idx = t_user_list.index(user)
                t_item[item][idx] = 1.0
            else:
                t_item[item] = t_i_init
                idx = t_user_list.index(user)
                t_item[item][idx] = 1.0

        else:
            if user in t_user:
                idx = t_item_list.index(item)
                t_user[user][idx] = rating
            else:
                t_user[user] = t_u_init
                idx = t_item_list.index(item)
                t_user[user][idx] = rating
            if item in t_item:
                idx = t_user_list.index(user)
                t_item[item][idx] = 1.0
            else:
                t_item[item] = t_i_init
                idx = t_user_list.index(user)
                t_item[item][idx] = 1.0

            t_train.append([user, item, rating])

    f.close()

    return t_user, t_item, t_user_list, t_item_list
