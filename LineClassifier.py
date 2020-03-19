import  numpy as np
import math
import ReadDataSet as data
from matplotlib import pyplot as plt
def sigmoid(f):
    return 1/(1+np.exp(-f))
def loss_compute(Y,Z):
    loss=-(np.multiply(Y, np.log(Z))+np.multiply(1-Y, np.log(1-Z)))
    loss=np.mean(loss)
    return loss
def line_f(W,b,X,Y):

    #X=batch X 4    W=4X  b=1x
    f=np.matmul(X,W)+b
    z=sigmoid(f)
    loss=loss_compute(Y,z)
    dW = np.multiply(z-Y,X.T)
    dW = np.mean(dW,axis=1)
    db = np.mean(z-Y)
    return loss,dW,db

def test(W,b,val_feature, val_label):
    f = np.matmul(val_feature, W) + b
    z = sigmoid(f)
    z = np.where(z >= 0.5, 1, 0)
    accuracy =np.sum(z == val_label) / len(val_label)
    return accuracy

def visual(loss,number):
    plt.figure(figsize=(6, 4), dpi=120)
    plt.grid()
    plt.xlabel('The number of iterations in training model')
    plt.ylabel('Loss')
    plt.plot(number, loss, label='Loss')
    plt.legend()
    plt.show()
def visual_acc(acc,number):
    plt.figure(figsize=(6, 4), dpi=120)
    plt.grid()
    plt.xlabel('The number of iterations in training model')
    plt.ylabel('test_accuracy')
    plt.plot(number, acc, label='test_accuracy')
    plt.legend()
    plt.show()

def train(train_feature,train_label,val_feature, val_label,learning_rate=0.001):

    W = np.random.random_sample(4,)*0.01
    b = np.random.random_sample(1,)*0.01
    X=train_feature
    Y=train_label
    id=0
    epcos=[]
    accs=[]
    losses=[]
    for epco in range(3000):

        start=id*5
        end=id*5+5
        if end>=500:
            id=0
        loss, dW, db = line_f(W, b, X[start:end,:], Y[start:end])
        W = W - learning_rate*dW
        b = b - learning_rate * db
        acc=test(W,b,val_feature, val_label)
        epcos.append(epco)
        accs.append(acc)
        losses.append(loss)
        id+=1
    return W,b,epcos,accs,losses


if __name__ == '__main__':
    avg_age=data.pre_process_age("dataset.txt")
    feature_list_all,gt_label_list_all=data.read_data("dataset.txt", avg_age)
    #测试5次，每次随机选取500个样本作为训练集，50个样本作为测试集
    mean_acc=0.0
    for i in range(5):

           train_feature,train_label,val_feature,val_label=data.getTrainVal(feature_list_all, gt_label_list_all)
           mean= np.mean(train_feature, axis=0)

           train_feature=train_feature-mean
           val_feature=val_feature-mean
           W,b,epcos,accs,losses=train(train_feature, train_label,val_feature, val_label,learning_rate=0.003)
           #visual_acc(accs,epcos)
           accuracy=test(W, b, val_feature, val_label)
           mean_acc+=accuracy
           print("Test_",i+1," accuracy:",accuracy)
    print("Test_mean"," accuracy:", mean_acc/5)
