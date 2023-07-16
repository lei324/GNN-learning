import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib

matplotlib.rc('font',family='SimHei')
plt.rcParams['axes.unicode_minus']=False
plt.plot([1,2,3],[100,500,200])
plt.title("文字测试",fontsize=25)
plt.xlabel('X',fontsize=15)
plt.ylabel('Y',fontsize=15)
plt.show()