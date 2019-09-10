import numpy as np
import sonnet as snt
import tensorflow as tf
from sklearn import tree
import copy
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


skew_rank = accu_rank = [24, 39, 2, 21, 31, 36, 19, 20, 18, 33, 25, 27, 1, 6, 8, 7, 32, 3, 11, 34, 5, 9, 12, 37, 23, 0, 28, 38, 29, 15, 16, 13, 30, 10, 35, 14, 26, 17, 22, 4]

def load_celeba(data_dir, restricted_degree, label_type, print_ratio=False):
    """Returns CelebA as (train_data, train_labels, test_data, test_labels)

        Shapes are (162770, 64, 64, 3), (162770, 2), (19962, 64, 64, 3), (19962, 10)
        Data is in [0,1] and labels are one-hot

        Arg:
          restricted_degree: only keep the instances with at least d selected attributes
    """
    train_data = np.load(os.path.join(data_dir, 'celeba_train_imgs.npy'))
    test_data = np.load(os.path.join(data_dir, 'celeba_test_imgs.npy'))

    info_pak = np.load(os.path.join(data_dir, 'celeba_attr.npz'))
    train_idxs = info_pak['train_idxs']
    val_idxs = info_pak['val_idxs']
    test_idxs = info_pak['test_idxs']
    attribute_names = info_pak['attribute_names']
    attributes = info_pak['attributes']
    male_attr_idx = 20

    def get_label(data, idxs):
        def count_indicators(attr):
            important_attributes_idx = [0, 1, 4, 9, 16, 18, 22, 24, 29, 30, 34, 36, 37, 38]
            x = np.array([0] * attr.shape[0])
            for i in important_attributes_idx:
                x = x + attr[:, i]
            return x

        label = attributes[idxs]
        sig = count_indicators(label) >= restricted_degree
        label = label[sig]
        data = data[sig]

        if label_type == 'gender':
            label = 1-label[:, male_attr_idx].reshape([-1, 1])
            label = np.append(label, 1 - label, 1)
        elif label_type == 'subattr':
            # decission_tree_attr_idx = [1, 6, 34, 35, 36]
            # decission_tree_attr_idx = [0, 1, 6, 7, 8, 9, 12, 18, 19, male_attr_idx, 24, 34, 36, 38, 39]
            decission_tree_attr_idx = [i for i in range(label.shape[1])]
            sub_attributes_idx = np.array(decission_tree_attr_idx)
            label = label[:, sub_attributes_idx]
        return data, label

    train_data, train_label = get_label(train_data, train_idxs)
    test_data, test_label = get_label(test_data, test_idxs)

    if print_ratio:
        print('\nCelebA restricted degree: {}'.format(restricted_degree))
        train_ratio = sum(train_label[:, 1]) / train_label.shape[0]
        test_ratio = sum(test_label[:, 1]) / test_label.shape[0]
        print('Training set - Male: {:.2f}% ({}/{}), Not male: {:.2f}%'.format(train_ratio * 100,
                                                                               sum(train_label[:, 1]),
                                                                               train_label.shape[0],
                                                                               100 - train_ratio * 100))
        print('Testing set - Male: {:.2f}% ({}/{}), Not male: {:.2f}%'.format(test_ratio * 100,
                                                                              sum(test_label[:, 1]),
                                                                              test_label.shape[0],
                                                                              100 - test_ratio * 100))

    return train_data, train_label, test_data, test_label

train_data, train_labels, test_data, test_labels = load_celeba(
        'H:\\CodeRange\\CelebA\\npy\\', restricted_degree=0, print_ratio=False, label_type='gender')

_, train_latent_labels, _, test_latent_labels = load_celeba(
    'H:\\CodeRange\\CelebA\\npy\\', restricted_degree=0, print_ratio=False, label_type='subattr')

attributes_names = np.load('H:\\CodeRange\\CelebA\\npy\\celeba_attr.npz')['attribute_names']

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def work(skewRankIdx):
    path = './saved_model/SkewRank{:02d}_{}/'.format(skewRankIdx, attributes_names[skew_rank[skewRankIdx]])
    print(path)
    model=tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], path+'/saved_model')
    
    def get_variables(model_meta):
        graph = tf.get_default_graph()
        sig_vars = copy.deepcopy(model_meta.signature_def['serving_default'])
        sig_inputs = sig_vars.inputs
        sig_outputs = sig_vars.outputs
        output = dict()
        for k in sig_inputs.keys():
            print('{:20}, {}'.format(k,sig_inputs[k].name))
            output[k] = graph.get_tensor_by_name(sig_inputs[k].name)
        for k in sig_outputs.keys():
            print('{:20}, {}'.format(k,sig_outputs[k].name))
            output[k] = graph.get_tensor_by_name(sig_outputs[k].name)
        return output

    tensors = get_variables(model)

    t_x = tensors['x']
    t_latent_var = tensors['latent_var']
    t_output = tensors['output']
    
    
    
    epoch_size = 32

    def get_latent_var_list(data):
        test_data = data
        test_data_len = test_data.shape[0]
        epoch_num = test_data_len // epoch_size
        # epoch_num = 60
        instance_num = epoch_num * epoch_size

        latent_var = []
        for i in tqdm(range(epoch_num)):
            epoch_beg = i*epoch_size
            epoch_end = (i+1)*epoch_size
            epoch_input = test_data[epoch_beg:epoch_end].astype('float32') / 255.0
        #     print(epoch_beg, epoch_end)
            outputs = session.run([t_output, t_latent_var], 
                                  feed_dict={t_x:epoch_input})
            latent_var.append(outputs[1])
        return np.array(latent_var)

    train_latent_var_lists = get_latent_var_list(train_data)
    test_latent_var_lists = get_latent_var_list(test_data)
    
    np.save(os.path.join(path, 'post_latent_var-train.npy'), train_latent_var_lists)
    np.save(os.path.join(path, 'post_latent_var-test.npy'), test_latent_var_lists)
    
for i in range(40):
    work(i)
