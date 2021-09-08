import pandas as pd
import numpy as np

import torch

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import seaborn as sn
import matplotlib.pyplot as plt

def cat2player(cat):
    c2p = {0:'ANTONSEN', 1:'GINTING', 2:'Long', 3:'Yufei', 4:'CHOU', 5:'CHRISTIE',
           6:'MOMOTA', 7:'PHETPRADAB', 8:'NG', 9:'PUSARLA', 10:'SHI', 11:'TAI',
           12:'AXELSEN', 13:'WANG'}
    return c2p[cat]

# def cat2player(cat):
#     c2p = {0:'ANTONSEN', 1:'GINTING', 2:'CHOU',
#        3:'CHRISTIE', 4:'MOMOTA', 5:'NG',
#        6:'AXELSEN'}
#     return c2p[cat]

def metrices(preds, labels, num_classes):
    acc = []
    precision = []
    recall = []
    f1 = []
    count = [] 
    
    TP_all = 0
    FP_all = 0
    TN_all = 0
    FN_all = 0
    for target in range(num_classes):
        cnt = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(preds)):
            if labels[i]==target:
                cnt += 1
            if preds[i]==target and labels[i]==target:
                TP += 1
                TP_all += 1
            elif preds[i]==target and labels[i]!=target:
                FP += 1
                FP_all += 1
            elif preds[i]!=target and labels[i]!=target:
                TN += 1
                TN_all += 1
            elif preds[i]!=target and labels[i]==target:
                FN += 1
                FN_all += 1
        acc.append((TP+TN)/(TP+FP+TN+FN+1e-10))
        precision.append((TP)/(TP+FP+1e-10))
        recall.append(TP/(TP+FN+1e-10))
        f1.append(2*(precision[target]*recall[target])/(precision[target]+recall[target]+1e-10))
        count.append(cnt)
            
    num_exist_class = len([1 for c in count if c!=0]) + 1e-10
    return sum(acc)/num_exist_class, sum(precision)/num_exist_class, sum(recall)/num_exist_class, sum(f1)/num_exist_class, None

def confusion_matrix(preds, labels, num_classes, ax):
    conf = np.zeros((num_classes, num_classes))
    for i in range(len(preds)):
        conf[preds[i]][labels[i]] += 1

    norm_vec = np.sum(conf, axis=0)
    conf = np.around(conf/norm_vec, decimals=2)
    df_cm = pd.DataFrame(conf, [cat2player(i) for i in range(num_classes)], [cat2player(i) for i in range(num_classes)])
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap='GnBu', fmt='g', ax=ax)
    b, t = ax.get_ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    ax.set_title("Confusion Matrix of Player Classification")
    ax.set_ylim(b, t) # update the ylim(bottom, top) values
    ax.xaxis.set_tick_params(rotation=45)
    ax.yaxis.set_tick_params(rotation=45)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Prediction')
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
    ax.title.set_fontsize(16)

def moving_average(lst, wind_size):
    lst = [sum(lst[i:i+wind_size])/wind_size for i in range(0, len(lst)-wind_size)]
    return lst

def plots(losses, accs, f1s, wind_size, ax1):
    ax2 = ax1.twinx()
    
    y1 = moving_average(losses, wind_size)
    x1 = [i for i in range(len(y1))]
    
    y2 = accs
    x2_step = round(len(x1)/len(y2))
    x2 = [i*x2_step for i in range(1, len(y2)+1)]
    
    y3 = f1s

    curve1, = ax1.plot(x1, y1, label="Training loss", color='r')
    curve2, = ax2.plot(x2, y2, label="Test accuracy", color='b')
    curve3, = ax2.plot(x2, y3, label="Test F1", color='g')
    
    curves = [curve1, curve2, curve3]
    ax1.legend(curves, [curve.get_label() for curve in curves], loc='center right')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Acc/F1")
    
    ax1.set_title("Learning Curve")
    
    for item in ([ax1.xaxis.label, ax1.yaxis.label, ax2.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels() +  ax2.get_yticklabels()):
        item.set_fontsize(12)
    ax1.title.set_fontsize(16)

def tsne_viz_train_test(train_embeddings, train_labels, test_embeddings, test_labels, class_embedding, ax, mode):
    train_range = train_embeddings.shape[0]
    test_range = train_range + test_embeddings.shape[0]
    embeddings = np.concatenate((train_embeddings, test_embeddings, class_embedding), axis=0)
    labels = np.concatenate((train_labels, test_labels, np.array([i for i in range(class_embedding.shape[0])])), axis=0)

    if mode=='tsne':
        reduced = TSNE(n_components=2, perplexity=30.0).fit_transform(embeddings)
    elif mode=='pca':
        reduced = PCA(n_components=2, svd_solver='full').fit_transform(embeddings)
    else:
        raise NotImplemented('Only t-SNE and PCA visualizations are supported currenly')
    
    train_x = reduced[:train_range, 0]
    train_y = reduced[:train_range, 1]
    train_group = labels[:train_range]
    
    test_x = reduced[train_range:test_range, 0]
    test_y = reduced[train_range:test_range, 1]
    test_group = labels[train_range:test_range]
    
    class_x = reduced[test_range:, 0]
    class_y = reduced[test_range:, 1]
    class_group = labels[test_range:]

    
    cdict = {0:'black', 1:'silver', 2:'lightcoral', 3:'red', 4:'sienna', 5:'orange',
           6:'yellow', 7:'green', 8:'lime', 9:'blue', 10:'cyan', 11:'purple',
           12:'lightblue', 13:'deeppink', 14:'darkmagenta', 15:'dodgerblue', 16:'lime'}

#     cdict = {0: 'red', 1: 'blue', 2: 'green', 3: 'cyan', 4: 'magenta', 5: 'yellow', 6: 'gray'}

    for g in np.unique(train_group):
        ix = np.where(train_group == g)
        ax.scatter(train_x[ix], train_y[ix], c = cdict[g], s = 4, alpha=0.05)
    
    for g in np.unique(test_group):
        ix = np.where(test_group == g)
        ax.scatter(test_x[ix], test_y[ix], c = cdict[g], label = cat2player(g), s = 10, alpha=0.8)
    
    for g in np.unique(class_group):
        ix = np.where(class_group == g)
        ax.scatter(class_x[ix], class_y[ix], c = cdict[g], s = 200, marker = '*', edgecolors='black')
    
    ax.legend()
    if mode == 'tsne':
        ax.set_title('Visualization of Embeddings (t-SNE)')
    else:
        ax.set_title('Visualization of Embeddings (PCA)')
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    
def class_distance_viz(class_embedding, num_classes):
    class_dist = torch.cdist(class_embedding, class_embedding, p=2).cpu().numpy()#/class_embedding.shape[1]
    fig, ax = plt.subplots(figsize=(7, 5))
    df = pd.DataFrame(class_dist, [cat2player(i) for i in range(num_classes)], [cat2player(i) for i in range(num_classes)])
    
    sn.heatmap(df, annot=True, annot_kws={"size": 10}, cmap='GnBu', fmt='.2f', ax=ax, square=True)
    b, t = ax.get_ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    ax.set_title("Distance of Embedding between Players")
    ax.set_ylim(b, t) # update the ylim(bottom, top) values
    ax.xaxis.set_tick_params(rotation=90)
    ax.yaxis.set_tick_params(rotation=0)
    ax.set_xlabel('Player')
    ax.set_ylabel('Player')
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
    ax.title.set_fontsize(16)