import json
import numpy as np
import re
from string import punctuation


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)

    return string.strip().lower()


def read_dataset(s_path, t_path):
    embed_dict, w_embed, s_user, s_item, t_user, t_item = dict(), dict(), dict(), dict(), dict(), dict()
    print('\nLoading Glove ... \n')

    f = open('../meta_learning/data/glove.6B.100d.txt')

    for line in f:
        word_vector = line.split()
        word = word_vector[0]
        word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
        w_embed[word] = word_vector_arr

    f.close()

    s_data, t_train, t_valid, t_test = [], [], [], []
    len_t_data = 0

    print('\nLoading data ... \n')

    f = open(s_path, 'r')
    s_user_list, s_item_list = [], []

    while True:
        # Read one line & Learn
        line = f.readline()
        if not line: break

        # Convert str to json format
        line = json.loads(line)

        s_user_list.append(line['reviewerID'])
        s_item_list.append(line['asin'])
        
    f.close()

    s_user_list = list(set(s_user_list))
    s_item_list = list(set(s_item_list))

    s_u_dim = len(s_item_list)
    s_i_dim = len(s_user_list)

    s_u_init = [0.0] * s_u_dim
    s_i_init = [0.0] * s_i_dim

    f = open(s_path, 'r')
    s_len = 0

    while True:
        s_len += 1
        if s_len % 10000 == 0:
            print(s_len)
            break

        # Read one line & Learn
        line = f.readline()
        if not line: break

        # Convert str to json format
        line = json.loads(line)

        user = line['reviewerID']
        item = line['asin']
        review = line['reviewText']
        rating = line['overall']

        if user in s_user:
            idx = s_item_list.index(item)
            s_user[user][idx] = rating
        else:
            s_user[user] = s_u_init
            idx = s_item_list.index(item)
            s_user[user][idx] = rating
        if item in s_item:
            idx = s_user_list.index(user)
            s_item[item][idx] = 1.0
        else:
            s_item[item] = s_i_init
            idx = s_user_list.index(user)
            s_item[item][idx] = 1.0

        review = review.lower()
        # Remove punctuation
        review = ''.join([c for c in review if c not in punctuation])
        review = review.split(' ')

        for rev in review:
            try:
                rev = clean_str(rev)
                rev = w_embed[rev]

                if user in embed_dict:
                    embed_dict[user].append(rev)
                else:
                    embed_dict[user] = [rev]

                if item in embed_dict:
                    embed_dict[item].append(rev)
                else:
                    embed_dict[item] = [rev]

            except KeyError:
                continue

        s_data.append([user, item, rating])

    f.close()

    t_user_list, t_item_list = [], []
    f = open(t_path, 'r')
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
        review = line['reviewText']
        rating = line['overall']

        review = review.lower()
        # Remove punctuation
        review = ''.join([c for c in review if c not in punctuation])
        review = review.split(' ')

        if user in embed_dict and item in embed_dict and len(t_valid) < len_t_data:
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
                t_item[item][idx] = 0.0
            else:
                t_item[item] = t_i_init
                idx = t_user_list.index(user)
                t_item[item][idx] = 0.0

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

            for rev in review:
                try:
                    rev = clean_str(rev)
                    rev = w_embed[rev]

                    if user in embed_dict:
                        embed_dict[user].append(rev)
                    else:
                        embed_dict[user] = [rev]

                    if item in embed_dict:
                        embed_dict[item].append(rev)
                    else:
                        embed_dict[item] = [rev]

                except KeyError:
                    continue

    f.close()

    t_test = t_valid[int(len_t_data/2):len_t_data]
    t_valid = t_valid[0:int(len_t_data/2)]

    print(len(s_data), len(embed_dict), len(t_train), len(t_valid), len(t_test))

    return s_data, t_train, t_valid, t_test, embed_dict, s_user, s_item, t_user, t_item, s_user_list, s_item_list, t_user_list, t_item_list


def read_yelp_dataset(s_path, t_path):
    embed_dict, w_embed, s_user, s_item, t_user, t_item = dict(), dict(), dict(), dict(), dict(), dict()
    print('\nLoading Glove ... \n')

    f = open('../meta_learning/data/glove.6B.100d.txt')

    for line in f:
        word_vector = line.split()
        word = word_vector[0]
        word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
        w_embed[word] = word_vector_arr

    f.close()

    s_data, t_train, t_valid, t_test = [], [], [], []
    len_t_data = 0

    print('\nLoading data ... \n')

    f = open(s_path, 'r')
    s_user_list, s_item_list = [], []

    lin = 0
    while True:
        lin += 1
        # Read one line & Learn
        line = f.readline()
        if not line or lin > 10000: break

        # Convert str to json format
        line = json.loads(line)

        s_user_list.append(line['user_id'])
        s_item_list.append(line['business_id'])
    f.close()

    s_user_list = list(set(s_user_list))
    s_item_list = list(set(s_item_list))

    s_u_dim = len(s_item_list)
    s_i_dim = len(s_user_list)

    s_u_init = [0.0] * s_u_dim
    s_i_init = [0.0] * s_i_dim

    f = open(s_path, 'r')
    s_len = 0

    lin = 0
    while True:
        s_len += 1
        lin += 1

        if s_len % 10000 == 0:
            print(s_len)

        # Read one line & Learn
        line = f.readline()
        if not line or lin > 10000: break

        # Convert str to json format
        line = json.loads(line)

        user = line['user_id']
        item = line['business_id']
        review = line['text']
        rating = line['stars']

        if user in s_user:
            idx = s_item_list.index(item)
            s_user[user][idx] = rating
        else:
            s_user[user] = s_u_init
            idx = s_item_list.index(item)
            s_user[user][idx] = rating
        if item in s_item:
            idx = s_user_list.index(user)
            s_item[item][idx] = 1.0
        else:
            s_item[item] = s_i_init
            idx = s_user_list.index(user)
            s_item[item][idx] = 1.0

        review = review.lower()
        # Remove punctuation
        review = ''.join([c for c in review if c not in punctuation])
        review = review.split(' ')

        for rev in review:
            try:
                rev = clean_str(rev)
                rev = w_embed[rev]

                if user in embed_dict:
                    embed_dict[user].append(rev)
                else:
                    embed_dict[user] = [rev]

                if item in embed_dict:
                    embed_dict[item].append(rev)
                else:
                    embed_dict[item] = [rev]

            except KeyError:
                continue

        s_data.append([user, item, rating])

    f.close()

    t_user_list, t_item_list = [], []
    f = open(t_path, 'r')
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
        review = line['reviewText']
        rating = line['overall']

        review = review.lower()
        # Remove punctuation
        review = ''.join([c for c in review if c not in punctuation])
        review = review.split(' ')

        if user in embed_dict and item in embed_dict and len(t_valid) < len_t_data:
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
                t_item[item][idx] = 0.0
            else:
                t_item[item] = t_i_init
                idx = t_user_list.index(user)
                t_item[item][idx] = 0.0

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

            for rev in review:
                try:
                    rev = clean_str(rev)
                    rev = w_embed[rev]

                    if user in embed_dict:
                        embed_dict[user].append(rev)
                    else:
                        embed_dict[user] = [rev]

                    if item in embed_dict:
                        embed_dict[item].append(rev)
                    else:
                        embed_dict[item] = [rev]

                except KeyError:
                    continue

    f.close()

    t_test = t_valid[int(len_t_data/2):len_t_data]
    t_valid = t_valid[0:int(len_t_data/2)]

    print(len(embed_dict), len(t_train), len(t_valid), len(t_test))

    return s_data, t_train, t_valid, t_test, embed_dict, s_user, s_item, t_user, t_item, s_user_list, s_item_list, t_user_list, t_item_list


def read_new_dataset(s_path, t_path):
    embed_dict, w_embed, s_user, s_item, t_user, t_item = dict(), dict(), dict(), dict(), dict(), dict()
    print('\nLoading Glove ... \n')

    f = open('../meta_learning/data/glove.6B.100d.txt')

    for line in f:
        word_vector = line.split()
        word = word_vector[0]
        word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
        w_embed[word] = word_vector_arr

    f.close()

    s_data, t_train, t_valid, t_test = [], [], [], []
    len_t_data = 0

    print('\nLoading data ... \n')

    f = open(s_path, 'r')
    s_user_list, s_item_list = [], []

    while True:
        # Read one line & Learn
        line = f.readline()
        if not line: break

        # Convert str to json format
        line = json.loads(line)

        s_user_list.append(line['reviewerID'])
        s_item_list.append(line['asin'])
    f.close()

    s_user_list = list(set(s_user_list))
    s_item_list = list(set(s_item_list))

    s_u_dim = len(s_item_list)
    s_i_dim = len(s_user_list)

    s_u_init = [0.0] * s_u_dim
    s_i_init = [0.0] * s_i_dim

    f = open(s_path, 'r')
    s_len = 0

    while True:
        s_len += 1
        if s_len % 10000 == 0:
            print(s_len)

        # Read one line & Learn
        line = f.readline()
        if not line: break

        # Convert str to json format
        line = json.loads(line)

        user = line['reviewerID']
        item = line['asin']
        review = line['reviewText']
        rating = line['overall']

        if user in s_user:
            idx = s_item_list.index(item)
            s_user[user][idx] = rating
        else:
            s_user[user] = s_u_init
            idx = s_item_list.index(item)
            s_user[user][idx] = rating
        if item in s_item:
            idx = s_user_list.index(user)
            s_item[item][idx] = 1.0
        else:
            s_item[item] = s_i_init
            idx = s_user_list.index(user)
            s_item[item][idx] = 1.0

        review = review.lower()
        # Remove punctuation
        review = ''.join([c for c in review if c not in punctuation])
        review = review.split(' ')

        for rev in review:
            try:
                rev = clean_str(rev)
                rev = w_embed[rev]

                if user in embed_dict:
                    embed_dict[user].append([item, rev])
                else:
                    embed_dict[user] = [[item, rev]]

                if item in embed_dict:
                    embed_dict[item].append([user, rev])
                else:
                    embed_dict[item] = [[user, rev]]

            except KeyError:
                continue

        s_data.append([user, item, rating])

    f.close()

    t_user_list, t_item_list = [], []
    f = open(t_path, 'r')
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
        review = line['reviewText']
        rating = line['overall']

        review = review.lower()
        # Remove punctuation
        review = ''.join([c for c in review if c not in punctuation])
        review = review.split(' ')

        if user in embed_dict and item in embed_dict and len(t_valid) < len_t_data:
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
                t_item[item][idx] = 0.0
            else:
                t_item[item] = t_i_init
                idx = t_user_list.index(user)
                t_item[item][idx] = 0.0

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

            for rev in review:
                try:
                    rev = clean_str(rev)
                    rev = w_embed[rev]

                    if user in embed_dict:
                        embed_dict[user].append([item, rev])
                    else:
                        embed_dict[user] = [[item, rev]]

                    if item in embed_dict:
                        embed_dict[item].append([user, rev])
                    else:
                        embed_dict[item] = [[user, rev]]

                except KeyError:
                    continue

    f.close()

    t_test = t_valid[int(len_t_data/2):len_t_data]
    t_valid = t_valid[0:int(len_t_data/2)]

    print(len(embed_dict), len(t_train), len(t_valid), len(t_test))

    return s_data, t_train, t_valid, t_test, embed_dict, s_user, s_item, t_user, t_item, s_user_list, s_item_list, t_user_list, t_item_list


def read_mmt_dataset(t_path):
    embed_dict, w_embed, t_user, t_item = dict(), dict(), dict(), dict()
    print('\nLoading Glove ... \n')

    f = open('../meta_learning/data/glove.6B.100d.txt')

    for line in f:
        word_vector = line.split()
        word = word_vector[0]
        word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
        w_embed[word] = word_vector_arr

    f.close()

    s_data, t_train, t_valid, t_test = [], [], [], []
    len_t_data = 0

    print('\nLoading data ... \n')

    t_user_list, t_item_list = [], []
    f = open(t_path, 'r')
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
        review = line['reviewText']
        rating = line['overall']

        review = review.lower()
        # Remove punctuation
        review = ''.join([c for c in review if c not in punctuation])
        review = review.split(' ')

        for rev in review:
            try:
                rev = clean_str(rev)
                rev = w_embed[rev]

                if user + item in embed_dict:
                    embed_dict[user + item].append(rev)
                else:
                    embed_dict[user + item] = [rev]
                if user not in embed_dict:
                    embed_dict[user] = [0]
                if item not in embed_dict:
                    embed_dict[item] = [0]

            except KeyError:
                continue

        if user in embed_dict and item in embed_dict and len(t_valid) < len_t_data:
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
                t_item[item][idx] = 0.0
            else:
                t_item[item] = t_i_init
                idx = t_user_list.index(user)
                t_item[item][idx] = 0.0

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

    t_test = t_valid[int(len_t_data/2):len_t_data]
    t_valid = t_valid[0:int(len_t_data/2)]

    print(len(embed_dict), len(t_train), len(t_valid), len(t_test))

    return t_train, t_valid, t_test, embed_dict, t_user, t_item, t_user_list, t_item_list