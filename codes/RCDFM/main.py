from train_rcdfm import learning, valid, ndcg_valid
from pro_rdcfm import read_dataset, read_yelp_dataset
import torch
from ndcg import read_targ_dataset, cal_ndcg

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    source_path = '../meta_learning/data/baby.json'
    #source_path = '../meta_learning/data/kindle.json'
    #source_path = '../meta_learning/data/tag.json'
    #source_path = '../meta_learning/data/yelp.json'

    #target_path = '../meta_learning/data/Automotive.json'
    target_path = '../meta_learning/data/Patio_Lawn_and_Garden.json'
    #target_path = '../meta_learning/data/Instant_Video.json'
    #target_path = '../meta_learning/data/Office.json'
    print(source_path, target_path)
    real, idcg_val, idcg = read_targ_dataset(target_path)

    path = 'ablation_' + source_path[22:-5] + '_' + target_path[22:-5]
    save = '../meta_learning/trained_model/rcdfm_' + path + '.pth'
    write_file = './result/rcdfm/' + path + '.txt'
    iteration = 200

    s_data, t_train, t_valid, t_test, user_dict, s_user, s_item, t_user, t_item, s_user_list, s_item_list, t_user_list, t_item_list = read_dataset(source_path, target_path)
    #s_data, t_train, t_valid, t_test, user_dict, s_user, s_item, t_user, t_item, s_user_list, s_item_list, t_user_list, t_item_list = read_yelp_dataset(source_path, target_path)

    for i in range(iteration):
        if i > 0:
            learning(s_data, t_train, user_dict, s_user, s_item, s_user_list, s_item_list, t_user, t_item, t_user_list, t_item_list, save, 1, device)
        else:
            learning(s_data, t_train, user_dict, s_user, s_item, s_user_list, s_item_list, t_user, t_item, t_user_list, t_item_list, save, 0, device)
            
        #valid(t_valid, t_test, user_dict, s_user, s_item, t_user, t_item, t_user_list, t_item_list, save, write_file, device)
        ndcg_mat = ndcg_valid(real, user_dict, s_user, s_item, t_user, t_item, t_user_list, t_item_list, save, write_file, device)
        cal_ndcg(ndcg_mat, idcg_val, idcg)
