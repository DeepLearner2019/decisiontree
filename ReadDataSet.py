import os.path as ops
import math
import cv2
import numpy as np
import random
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
                feature_list.append(float(avg_age))
            elif float(info_tmp[4]) < 1:
                feature_list.append(float(avg_age))
            else:
                feature_list.append(float(info_tmp[4]))

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

if __name__ == '__main__':
    avg_age=pre_process_age("dataset.txt")
    feature_list_all,gt_label_list_all=read_data("dataset.txt", avg_age)
    train_feature,train_label,val_feature,val_label=getTrainVal(feature_list_all, gt_label_list_all)
    print(train_feature.shape)
    print(train_label.shape)
    print(val_feature.shape)
    print(val_label.shape)