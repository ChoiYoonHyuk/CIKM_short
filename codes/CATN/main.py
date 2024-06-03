from data_reader import read_dataset, read_yelp_data
from train_catn import learning, valid
import torch

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    source_path = '../meta_learning/data/baby.json'
    #source_path = '../meta_learning/data/kindle.json'
    #source_path = '../meta_learning/data/tag.json'
    #source_path = '../meta_learning/data/yelp.json'

    #target_path = './dataset/fashion.json'
    target_path = './dataset/fashion_new.json'
    #target_path = '../meta_learning/data/Automotive.json'
    #target_path = '../meta_learning/data/Patio_Lawn_and_Garden.json'
    #target_path = '../meta_learning/data/Instant_Video.json'
    #target_path = '../meta_learning/data/Office.json'
    print(source_path, target_path)

    path = 'yelp_' + source_path[22:-5] + '_' + target_path[22:-5]
    save = '../meta_learning/trained_model/' + path + '.pth'
    write_file = './result/catn/' + path + '.txt'
    iteration = 100

    s_data, s_dict, t_train, t_valid, t_test, t_dict, w_embed = read_dataset(source_path, target_path)
    #s_data, s_dict, t_train, t_valid, t_test, t_dict, w_embed = read_yelp_data(source_path, target_path)

    for i in range(iteration):
        if i > 0:
            learning(s_data, s_dict, t_train, t_dict, w_embed, save, 1, device)
        else:
            learning(s_data, s_dict, t_train, t_dict, w_embed, save, 0, device)

        valid(s_dict, t_valid, t_dict, w_embed, save, t_test, write_file, device)
