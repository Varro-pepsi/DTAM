import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import itertools
import spectral
import matplotlib.pyplot as plt
from scipy import io
import imageio
import os
import re
import torch
import numpy as np
import yaml
from PIL import Image
from sklearn.decomposition import PCA

def seed_worker(seed):
    torch.manual_seed(seed)#生成一个随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)#生成一个gpu随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features gt的坐标列表
    y = gt[indices].ravel() # classes 打平后的gt
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
    train_label = []
    test_label = []
    if mode == 'random':
        if train_size == 1:#全作为训练样本
            random.shuffle(X)
            train_indices = [list(t) for t in zip(*X)]
            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))#按列合并，将train的data排列整齐放置在train_set中
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt = []
            test_set = []
        else:
            train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y, random_state=23)#随机分类训练和测试样本
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))
            [test_label.append(i) for i in gt[tuple(test_indices)]]
            test_set = np.column_stack((test_indices[0],test_indices[1],test_label))

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt, train_set, test_set


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))
    
def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool_)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    #target = target[ignored_mask] -1
    # target = target[ignored_mask]
    # prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion_matrix"] = cm

    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    results["TPR"] = TPR
    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1_scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    results["prediction"] = prediction
    results["label"] = target

    return results

def show_results(results, vis, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1_scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion_matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion_matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1_scores"]
        kappa = results["Kappa"]

    #label_values = label_values[1:]
    vis.heatmap(cm, opts={'title': "Confusion_matrix", 
                          'marginbottom': 150,
                          'marginleft': 150,
                          'width': 500,
                          'height': 500,
                          'rownames': label_values, 'columnnames': label_values})
    text += "Confusion_matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.03f} +- {:.03f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.03f}%\n".format(accuracy)
    text += "---\n"

    text += "F1_scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.03f} +- {:.03f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.03f}\n".format(kappa)

    vis.text(text.replace('\n', '<br/>'))
    print(text)

def picshow_save(Yshow,color_name,pic_save,show=None):
    
    color = color_chart(color_name)
    a,b=Yshow.shape
    c = Image.new("RGB",(b,a))
    
    for i in range(b):
        for j in range(a):
            k=int(Yshow[j,i])
            bar=color[k,:]            
            c.putpixel([i,j],(bar[0],bar[1],bar[2]))
    c.save(pic_save)
    if show:        
        c.show()


def color_chart(test_name="color_7"):
    color_16=np.array([[0, 0, 0], 
                          [128, 128, 128], 
                          [0, 255, 0], 
                          [0, 255, 255], 
                          [0, 128, 0], 
                          [255, 0, 255], 
                          [255, 255, 0], 
                          [0, 0, 255], 
                          [255, 0, 0], 
                          [128, 0, 0],
                          [0, 0, 128], 
                          [237, 145, 33], 
                          [221, 160, 221], 
                          [156, 102, 31], 
                          [255, 127, 80], 
                          [51, 161, 201], 
                          [139, 69, 19]])

    Hyrank_color=np.array([[0, 0, 0], 
                          [0, 0, 223], 
                          [0, 54, 255], 
                          [0, 146, 255], 
                          [0, 223, 255], 
                          [47, 255, 207], 
                          [143, 255, 111], 
                          [223, 255, 31], 
                          [255, 207, 0], 
                          [255, 113, 0],
                          [255, 31, 0], 
                          [207, 0, 0], 
                          [127, 0, 0]])
    
    color_7=np.array([[0, 0, 0], 
                          [0, 31, 255], 
                          [0, 175, 255], 
                          [63, 255, 191], 
                          [219, 255, 41], 
                          [255, 159, 0], 
                          [255, 48, 205], 
                          [255, 15, 0], 
                          [127, 0, 0], 
                          [206, 206, 128]])      
    if test_name == "color_7":
        color = color_7

    elif test_name == "color_16":
        color = color_16

    elif test_name == "color_12":
        color = Hyrank_color
    else: 
        raise ValueError("colorname_fail")

    return color

def evaluate(net, val_loader,gt,device, tgt=False, file=None):
    ps = []
    ys = []
    for i,(x1, y1) in enumerate(val_loader):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(device)
            p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)
    Num1=len(ys)
    Nc=0
    C=int(max(ys))#GT里面的有多少类
    for c in range(C):#按照每一类进行循环
        nc=0#GT每一类的个数
        ncc=0#预测中每一类的个数
        i=0#每一次都从0开始循环
        c=c+1
        index_c=ys==c
        index_PC=ps==c
        for i in range(Num1):    
            if index_c[i]==1: 
                nc=nc+1
            if index_PC[i]==1:           
                ncc=ncc+1
        Nc=Nc+nc*ncc#计算pe的分子
    pe=Nc/(Num1*Num1)#计算pe值
    Kappa=(acc-pe)/(1-pe)
    outputs = np.zeros((gt.shape[0],gt.shape[1]))
    n = 0
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]): 
                if int(gt[i,j]) == 0:
                    continue
                else :
                    outputs[i][j] = ps[n]+1
                    n+=1
        # if i % 200 == 0:
        #     print('... ... row ', i, ' handling ... ...')
    
    if tgt:
        results = metrics(ps, ys, n_classes=ys.max()+1)
        print(results['Confusion_matrix'],'\n','TPR:', np.round(results['TPR']*100,2),'\n', 'OA:', results['Accuracy'],file=file)
    return round((acc)*100,2),round(Kappa,4), outputs


def evaluate_tgt(cls_net, loader, gt, modelpath, device):
    cls_net.load_state_dict(torch.load(modelpath))
    cls_net.eval()
    acc,kappa,outputs = evaluate(cls_net, loader, gt,device, tgt=True)
    return acc,kappa,outputs

def apply_PCA(data, num_components=75):
    new_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_data = pca.fit_transform(new_data)
    new_data = np.reshape(new_data, (data.shape[0], data.shape[1], num_components))
    return new_data

def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict保存为yaml"""
    with open(save_path, 'w', encoding='utf-8') as file:
        yaml.dump(dict_value, file, encoding='utf-8', allow_unicode=True)

def read_yaml_to_dict(yaml_path: str, ):
    with open(yaml_path) as file:
        dict_value = yaml.safe_load(file)
        return dict_value

def set_config(current_path, parser, rask):
    config_path = os.path.join(current_path, 'config', f'{rask}_config.yaml')
    print(f'config路径是{config_path}')
    if not os.path.exists(config_path):#如果root不存在
        raise ValueError('config文件不存在')
    default_arg = read_yaml_to_dict(config_path)
    parser.set_defaults(**default_arg)
    return parser


# def set_config(current_path, parser):
#     config_path = os.path.join(current_path, 'config', f'config.yaml')
#     print(f'config路径是{config_path}')
#     if not os.path.exists(config_path):#如果root不存在
#         raise ValueError('config文件不存在')
#     default_arg = read_yaml_to_dict(config_path)
#     parser.set_defaults(**default_arg)
#     return parser


