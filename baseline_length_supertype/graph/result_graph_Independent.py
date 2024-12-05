import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FixedLocator
import matplotlib.colors as mcolors
import json



def sub_dataset_graph_more(ax,data,mg_hla_data):
    methods = ['Ann', 'pickpocket', 'Consensus', 'Smm', 'Smmpmbec','ACME','Anthem']
    metrics = ['AUC', 'ACC', 'MCC', 'F1']
    
    gap = 0.013  # 方法之间没有间隙
    color_palette = sns.color_palette("Purples", n_colors=len(methods))
    method_colors = {method: color for method, color in zip(methods , color_palette)}


    # 为每个性能指标设置基础位置
    x = np.arange(len(metrics)) * 1.5  # 增加间隔以清晰展示

    # 绘制每种方法的性能指标条形图
    bar_width = 0.14
    for i, method in enumerate(methods):
        ax.bar(x + i * (bar_width+gap), data[i], bar_width, label=method, color=method_colors[method],edgecolor='black', alpha=0.7)
        if i==0:
            ax.bar(x + i * (bar_width+gap), mg_hla_data[i] - data[i], bar_width,
            bottom=data[i], facecolor='none', edgecolor='black', linestyle='--', alpha=0.7, label=' MGHLA',capstyle='round')
        else:
            ax.bar(x + i * (bar_width+gap), mg_hla_data[i] - data[i], bar_width,
            bottom=data[i], facecolor='none', edgecolor='black', linestyle='--', alpha=0.7, capstyle='round')
    
    # 绘制虚线分隔不同的性能指标
    for i in range(1, len(metrics)):
        ax.axvline(x[i] - 3 * bar_width, color='gray', linestyle='--', alpha=0.7)



    # 设置坐标轴和图例
    ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value',fontsize=18)
    ax.set_ylim(45, 100)
    ax.set_title('Independent-subset Metrics',fontsize=20)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)


    plt.show()
    plt.savefig('Independent-subset Metrics.pdf',bbox_inches="tight")
    plt.savefig('Independent-subset Metrics.svg',bbox_inches="tight")


#总数据集画图
def all_data_graph(ax,performance_data):
    performance_data=performance_data*100
    # Methods and their corresponding performance metrics
    methods = ['netmhcpan_ba', 'netmhcpan_el', 'netmhcstabpan','HLAB', 'TransPHLA','MGHLA']
    

    metrics = ['AUC', 'ACC', 'MCC', 'F1']
    color_palette = sns.color_palette("Blues", n_colors=len(methods))
    method_colors = {method: color for method, color in zip(methods , color_palette)}

    gap = 0.013  # 方法之间没有间隙
    bar_width = 0.1  # Width of the bars
    index = np.arange(len(metrics))  # Metric indices
    for i in range(len(methods)):
        #plt.bar(index + i * bar_width, performance_data[i], bar_width, label=methods[i], color=highlight_color if methods[i] == 'MGHLA' else colors[i])
        plt.bar(index + i * (bar_width+gap), performance_data[i], bar_width, label=methods[i],edgecolor='black', color=method_colors[methods[i]],capstyle='round')

    # Adding labels and title
    plt.ylabel('Value', fontsize=18)
    plt.title('Independent Metrics', fontsize=20)
    plt.xticks(index + bar_width * (len(methods) - 1) / 2, metrics)
    plt.ylim(60, 100)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    

    # Adding a legend
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=4)

    # Show the plot
    #plt.tight_layout()
    plt.show()
    #plt.savefig('Independent Metrics.png',bbox_inches="tight")
    plt.savefig('Independent Metrics.pdf',bbox_inches="tight")
    plt.savefig('Independent Metrics.svg',bbox_inches="tight")



    
def select_colors_for_methods_seaborn():
    # 使用 Seaborn 生成13种颜色
    np.random.seed(4)
    color_hex=['darkgray','darkorange','salmon','chocolate','moccasin','darkseagreen','cadetblue','skyblue','lime','cornflowerblue','lightpink','thistle','orangered']
    # 定义13种方法
    methods = [
        "Ann", "Consensus", "Smm", "Smmpmbec",
        "netmhcpan_el", "netmhcpan_ba", "netmhcstabpan", "pickpocket",
        "ACME", "Anthem", "HLAB", "TransPHLA", "MGHLA"
    ]
   
    # 创建方法与颜色的对应字典
    method_color_mapping = dict(zip(methods, color_hex))
    
    # 保存对应关系到JSON文件，以便以后使用
    with open('method_color_mapping_seaborn.json', 'w') as f:
        json.dump(method_color_mapping, f, indent=4)
    
    return method_color_mapping

    
if __name__ == '__main__':
    
    mapping = select_colors_for_methods_seaborn()
    print(mapping)
    
    fig0, ax0 = plt.subplots(figsize=(10, 6))
    #对independent的全部数据方法作图
    performance_data = np.array([
        
        [0.9552, 0.802, 0.6496, 0.7579],   # Netmhcpan_ba
        [0.9568, 0.7952, 0.6433, 0.7449],   # Netmhcpan_el
        [0.9162, 0.7901, 0.6222, 0.7445],   # netmhcstabpan
        [0.93601, 0.93602, 0.8721, 0.9364], # HLAB
        [0.9778, 0.9294, 0.8588, 0.9298] ,# TransPHLA
        [0.9867,0.9466,0.8931,0.9468]
        
    ])
    #all_data_graph(performance_data)
    all_data_graph(ax0,performance_data)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    mapping = select_colors_for_methods_seaborn()
    print(mapping)

    data = np.array([
        [93.47, 78.12, 61.54, 72.63],  # Ann
        [93.4, 70.33, 49.46, 58.66],   # Pickpocket
        [93.21, 78.63, 62.39, 73.41],  # Consensus
        [91.05, 78.99, 60.74, 75.38],  # SMM
        [91.37, 78.9, 60.8, 75.38],     # SMMPMBEC
        [94.77, 78.99, 63.12, 73.88],#ACME
        [97.69, 91.39, 83.32, 90.88]   # Anthem
    ])

    mg_hla_data = np.array([
        [98.48, 94.21, 88.41, 94.23],  # MGHLA-ANN
        [98.80, 95.10, 90.20, 95.12],  # MGHLA-Pickpocket
        [98.48, 94.21, 88.42, 94.23],  # MGHLA-Consensus
        [98.65, 94.71, 89.42, 94.74],  # MGHLA-SMM
        [98.65, 94.72, 89.43, 94.75],   # MGHLA-SMMPMBEC
        [98.62, 94.50, 88.99, 94.52],
        [98.67,94.66,89.33,94.69] #MGHLA_Anthem
    ])
    
    

    # 调用绘图函数
    sub_dataset_graph_more(ax1,data,mg_hla_data)

    
    # 创建一个空白的axes仅用于显示图例
    
    # 提取图例
    handles0, labels0 = ax0.get_legend_handles_labels()
    handles1, labels1 =ax1.get_legend_handles_labels()
    handles = handles0 + handles1
    labels = labels0 + labels1
    # 创建一个新的空白图形
    ncol = len(handles) // 2
    fig_leg = plt.figure(figsize=(10, 1))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.legend(handles, labels, loc='center', ncol=ncol,frameon=False)
    ax_leg.axis('off')  # 隐藏坐标轴

    # 保存图例
    fig_leg.savefig('legend1.pdf', dpi=300, bbox_inches='tight', transparent=True)
    fig_leg.savefig('legend1.svg', dpi=300, bbox_inches='tight', transparent=True)

    
    
   