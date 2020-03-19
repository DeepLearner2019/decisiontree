import numpy as np
import os.path as ops
import random
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt

import pydotplus
from sklearn import tree
def pre_process_age(dataset_info_file):
    assert ops.exists(dataset_info_file), '{:s}　不存在'.format(dataset_info_file)
    sum=0.0
    num=0
    with open(dataset_info_file, 'r') as file:
        for _info in file:
            j = 0
            info_tmp=[]
            _info=_info.replace("\n", "")
            for i in range(len(_info)):
                if _info[i]==',':
                   #print(_info[j:i])
                   info_tmp.append(_info[j:i])
                   j=i+1
            #print(_info[j:])
            info_tmp.append(_info[j:])

            if info_tmp[4]=="NA":
                continue
            if float(info_tmp[4])>=1:
                sum+=float(info_tmp[4])
                num+=1
    return int(sum/num)

def read_data(dataset_info_file,avg_age):
    gt_label_list_all=[]
    feature_list_all=[]

    assert ops.exists(dataset_info_file), '{:s}　不存在'.format(dataset_info_file)

    with open(dataset_info_file, 'r') as file:
        for _info in file:#row
            j = 0
            info_tmp = []
            _info = _info.replace("\n", "")
            for i in range(len(_info)):
                if _info[i] == ',':
                    # print(_info[j:i])
                    info_tmp.append(_info[j:i])
                    j = i + 1
            # print(_info[j:])
            info_tmp.append(_info[j:])
            feature_list=[]

            gt_label_list_all.append(int(info_tmp[2]))

            if info_tmp[1] == "\"1st\"":
                feature_list.append(1)
            elif info_tmp[1] == "\"2nd\"":
                feature_list.append(2)
            else:
                feature_list.append(3)

            if info_tmp[4] == "NA":
                feature_list.append(int(float(avg_age))//10)
            elif float(info_tmp[4]) < 1:
                feature_list.append(int(float(avg_age))//10)
            else:
                feature_list.append(int(float(info_tmp[4]))//10)

            if info_tmp[5] == "\"Southampton\"":
                feature_list.append(1)
            elif info_tmp[5] == "\"Cherbourg\"":
                feature_list.append(2)
            elif info_tmp[5] == "\"Queenstown\"":
                feature_list.append(3)

            if info_tmp[6] == "\"male\"":
                feature_list.append(1)
            elif info_tmp[6] == "\"female\"":
                feature_list.append(2)
            feature_list_all.append(feature_list)


    random_idx = np.random.permutation(len(feature_list_all))
    feature_list = []
    label_list = []

    for index in random_idx:
        feature_list.append(feature_list_all[index])
        label_list.append(gt_label_list_all[index])
    feature_list_all = np.array(feature_list, np.float32)
    gt_label_list_all = np.array(label_list, np.float32)
    return feature_list_all,gt_label_list_all

def getTrainVal(feature_list_all,gt_label_list_all):
    index=random.randint(0, 500)
    val_feature=feature_list_all[index:index+50,:]
    val_label=gt_label_list_all[index:index+50]
    train_feature=np.concatenate([feature_list_all[0:index,:],feature_list_all[index+50:,:]],axis=0)
    train_label = np.concatenate([gt_label_list_all[0:index], gt_label_list_all[index + 50:]], axis=0)
    return train_feature,train_label,val_feature,val_label



def mytree(x_train, y_train, x_test, y_test, depth, is_pruning=True):
    if is_pruning:
        trees = DecisionTreeClassifier(max_depth=depth, random_state=0)
    else:
        trees = DecisionTreeClassifier(random_state=1)
    trees.fit(x_train, y_train)
    # 6、对测试数据进行预测，准确度较低，说明过拟合
    answer = trees.predict(x_test)
    y_test = y_test.reshape(-1)

    acc = np.mean(answer == y_test)
    #可视化决策树
    dot_data = tree.export_graphviz(trees, out_file=None,
                                    feature_names=['x1', 'x2','x3','x4'],
                                    class_names=['0', '1'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    with open('tree.png', 'wb') as f:
        f.write(graph.create_png())

    return acc
def visual():
    avg_age = pre_process_age("dataset.txt")
    feature_list_all, gt_label_list_all = read_data("dataset.txt", avg_age)
    train_feature, train_label, val_feature, val_label = getTrainVal(feature_list_all, gt_label_list_all)

    train_data = train_feature
    train_label = train_label

    test_data = val_feature
    test_label = val_label
    number = []
    accs = []
    for i in range(1, 13):
        acc = mytree(train_data, train_label, test_data, test_label, i, is_pruning=True)
        accs.append(acc)
        number.append(i)
    plt.figure(figsize=(6, 4), dpi=120)
    plt.grid()
    plt.xlabel('The max_depth of decision tree in training model')
    plt.ylabel('test_accuracy')
    plt.plot(number, accs, label='test_accuracy')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    avg_age = pre_process_age("dataset.txt")
    feature_list_all, gt_label_list_all = read_data("dataset.txt", avg_age)
    mean_acc=0.0
    for i in range(5):
         train_feature, train_label, val_feature, val_label = getTrainVal(feature_list_all, gt_label_list_all)
         accuracy = mytree(train_feature, train_label, val_feature, val_label, 3, is_pruning=True)
         mean_acc += accuracy
         print("Test_", i + 1, " accuracy:", accuracy)
print("Test_mean", " accuracy:", mean_acc / 5)
