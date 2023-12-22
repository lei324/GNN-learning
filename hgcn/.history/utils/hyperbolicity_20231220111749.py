# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    hyperbolicity.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: lei324 <lei324>                            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/20 10:22:51 by lei324            #+#    #+#              #
#    Updated: 2023/12/20 10:22:51 by lei324           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


# 随机采样 计算 Gromov 的 δ−双曲性 这是群论中的一个概念，用于衡量图的树形程度。 δ 越低，图数据集越双曲，对于树来说 δ = 0

import os
import pickle
import sys
import time

import networkx as nx
import numpy as np
from tqdm import tqdm

from utils.data_utils import load_data_lp


def hyperbolicity_sample(G, num_samples=50000):
    start_time = time.time()
    hyps = []
    for i in tqdm(range(num_samples)):
        start_time = time.time()
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        s = []
        try:
            d01 = nx.shortest_path_length(
                G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(
                G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(
                G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(
                G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(
                G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(
                G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01+d23)
            s.append(d02+d13)
            s.append(d03+d12)
            s.sort()
            hyps.append((s[-1]-s[-2])/2)
        except Exception as e:
            continue
    print('Time for hyp:', time.time()-start_time)

    return max(hyps)


if __name__ == "__main__":
    dataset = 'pubmed'
    data_path = os.path.join('./data', dataset)
    data = load_data_lp(dataset, use_feats=False, data_path=data_path)
    graph = nx.from_scipy_sparse_matrix(data['adj_train'])
    print('计算双曲系数：', graph.number_of_nodes(), graph.number_of_edges())
    hyp = hyperbolicity_sample(graph)
    print('Hyp:', hyp)
