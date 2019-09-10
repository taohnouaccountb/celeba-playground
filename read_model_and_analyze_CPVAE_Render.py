import numpy as np
from sklearn import tree
import copy
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from multiprocessing import Pool

data_dir = 'H:\\CodeRange\\CelebA\\npy\\'

info_pak = np.load(os.path.join(data_dir, 'celeba_attr.npz'))
train_idxs = info_pak['train_idxs']
val_idxs = info_pak['val_idxs']
test_idxs = info_pak['test_idxs']
attributes_names = info_pak['attribute_names']
attributes = info_pak['attributes']

train_label = attributes[train_idxs]
test_label = attributes[test_idxs]

skew_rank = [24, 39, 2, 21, 31, 36, 19, 20, 18, 33, 25, 27, 1, 6, 8, 7, 32, 3, 11, 34, 5, 9, 12, 37, 23, 0, 28, 38, 29, 15, 16, 13, 30, 10, 35, 14, 26, 17, 22, 4]

def add_info_to_node_label(in_filename, out_filename, node_id, info):
    with open(in_filename) as f:
        linelist = f.readlines()
    found_flag = False
    for i,s in enumerate(linelist):
        if s.startswith('{} [label='.format(node_id)):
            idx = s.find('", fillcolor=')
            s = s[:idx] + "\\n" + info + s[idx:]
            linelist[i] = s
            found_flag = True
            break
    if not found_flag:
        raise Exception('node {} not found'.format(node_id))
    with open(out_filename, "w+") as f:
        for x in linelist:
            f.write(x)

def tree2dot(decision_tree, path):
    tree.export_graphviz(
        decision_tree,
        out_file=path,
        class_names=['negative', 'positive'],
        filled=True,
        rounded=True,
        proportion=True)

def update_tree(skewRankIdx):
    initIdx = skew_rank[skewRankIdx]
    path = "./saved_model/SkewRank{:02d}_{}/".format(skewRankIdx, attributes_names[initIdx])
    train_latent_variable = np.reshape(np.load(os.path.join(path, 'post_latent_var-train.npy')), [-1, 50])
    test_latent_variable = np.reshape(np.load(os.path.join(path, 'post_latent_var-test.npy')), [-1, 50])
    with open(os.path.join(path, 'saved_model/decision_tree.pkl'), 'rb') as dt_file:
        decision_tree = pickle.load(dt_file)
    
    def draw_tree_for_class_other(class_other_idx):
        target_dot_path = os.path.join(path, 'tree__{}.dot'.format(attributes_names[class_other_idx]))
        target_png_path = os.path.join(path, 'tree_{}.png'.format(attributes_names[class_other_idx]))
        tree2dot(decision_tree, target_dot_path)
        
        def count(latent_variable, label):
            decision_path = decision_tree.decision_path(latent_variable).toarray()
            instances_num = decision_path.shape[0]
            nodes_num = decision_path.shape[1]

            node_pos = [0] * nodes_num
            node_neg = [0] * nodes_num
            print(nodes_num)
            for i in range(instances_num):
                for j in range(nodes_num):
                    if decision_path[i][j]:    
                        if label[i][class_other_idx]:
                            node_pos[j]+=1
                        else:
                            node_neg[j]+=1
            return node_pos, node_neg, nodes_num

        train_node_pos, train_node_neg, _ = count(train_latent_variable, train_label)
        test_node_pos, test_node_neg, nodes_num = count(test_latent_variable, test_label)
        for i in range(nodes_num):
            info = "train-Y_o[neg:pos]={}:{}\\ntest-Y_o[neg:pos]={}:{}".format(
                train_node_pos[i], train_node_neg[i], test_node_pos[i], test_node_neg[i])
            add_info_to_node_label(target_dot_path, target_dot_path, i, info)
            print(info)
        os.system("dot -Tpng {} -o {}".format(target_dot_path, target_png_path))
        os.remove(target_dot_path)
    for i in range(40):
        draw_tree_for_class_other(i)

for i in range(40):
    update_tree(i)