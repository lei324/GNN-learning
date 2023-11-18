import numpy as np
import scipy.sparse as sp
import torch

# 独热编码
def encode_onehot(labels):
    '''
    labels:输入一个类别标签
    output:return对应标签的独热编码数组
    '''
    classes=set(labels) #保证标签唯一
    classes_dict={c:np.identity(len(classes))[i,:] for i ,c in enumerate(classes)}
    labels_onehot=np.array(list(map(classes_dict.get,labels)),dtype=np.int32)
    return labels_onehot

#加载数据

def load_data(path='../data/cora/',dataset='cora'):
    print('Loading {} dataset...'.format(dataset))
    #读取结点数据
    idx_features_labels=np.genfromtxt("{}{}.content".format(path,dataset),dtype=np.dtype(str))
    #读取feature
    features=sp.csr_matrix(idx_features_labels[:,1:-1],dtype=np.float32)
    #读取标签
    labels = encode_onehot(idx_features_labels[:, -1])
    #节点序号
    idx=np.array(idx_features_labels[:,-1],dtype=np.int32)
    #节点map
    idx_map={j:i for i ,j in enumerate(idx)}
    
    #读取连接
    edges_unordered=np.genfromtxt('{}{}.cites'.format(path,dataset),dtype=np.int32)
    edges=np.array(list(map(idx_map.get,edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)  #原始是节点序号 现在变成从0 开始的
    #邻接矩阵 稀疏矩阵
    adj=sp.coo_matrix((np.ones(edges.shape[0]),(edges[:,0],edges[:,1])),shape=(labels.shape[0],labels.shape[0]),dtype=np.float32)

    #构建对称矩阵
    # 论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
    adj=adj+adj.T.multiply(adj.T > adj)-adj.multiply(adj.T > adj)   #与布尔矩阵相乘 得到大值

    features=normalize(features)
    adj=normalize(adj+sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features=torch.FloatTensor(np.array(features.todense()))
    labels=torch.LongTensor(np.where(labels)[1]) #即np.where(condition),只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。

    adj=sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    '''行均值化矩阵'''
    rowsum=np.array(mx.sum(1))
    r_inv=np.power(rowsum,-1).flatten()
    r_inv[np.isinf(r_inv)]=0
    r_mat_inv=sp.diags(r_inv)   #度矩阵  D
    mx=r_mat_inv.dot(mx)   #AD
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output,labels):
    preds=output.max(1)[1].type_as(labels)  #max(1) 延列方向的最大值 ，即每行的最大
    correct=preds.eq(labels).double()
    correct=correct.sum()
    return correct/len(labels)



    