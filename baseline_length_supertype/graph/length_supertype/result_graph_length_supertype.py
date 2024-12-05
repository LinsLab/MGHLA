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
import os
import ast
from collections import defaultdict

    
def parse_tuple(tuple_str):
    """
    手动解析元组字符串，将 'nan' 转换为 np.nan。
    """
    tuple_str = tuple_str.strip('()')
    elements = tuple_str.split(',')
    parsed_elements = []
    for elem in elements:
        elem = elem.strip()
        if elem.lower() == 'nan':
            parsed_elements.append(np.nan)
        else:
            try:
                parsed_elements.append(float(elem))
            except ValueError:
                parsed_elements.append(np.nan)
    return tuple(parsed_elements)
    
def read_and_parse_file(file_path):
    """
    读取文件并解析每行数据，返回一个字典。
    """
    data_dict = {}
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            # 找到第二个冒号的位置
            first_colon = line.find(':')
            if first_colon == -1:
                print(f"Line {line_number}: Error - Only one colon found.")
                continue
            second_colon = line.find(':', first_colon + 1)
            if second_colon == -1:
                print(f"Line {line_number}: Error - Only one colon found.")
                continue
            # 分割为键和值部分
            key_str = line[:second_colon]
            tuple_str = line[second_colon + 1:]
            try:
                value = parse_tuple(tuple_str)
            except Exception as e:
                print(f"Line {line_number}: Error parsing tuple: {e}")
                continue
            data_dict[key_str] = value
    return data_dict

def filter_keys_by_length(data_dict,method, target_length):
    """
    筛选包含特定长度值的键值对。

    Args:
        data_dict (dict): 原始数据字典。
        target_length (str 或 int): 需要筛选的长度值。

    Returns:
        dict: 筛选后的字典。
    """
    filtered_dict = {}
    target_length = str(target_length)
    for key, value in data_dict.items():
        if  '_{}_'.format(target_length) in key:
            if method+"_" in key or method+'i' in key:
                filtered_dict[key] = value
    return filtered_dict

def filter_keys_by_supertype(data_dict,method, supertype):
    print('筛选的超型是',supertype)
    filtered_dict = {}
    for key, value in data_dict.items():
        if  '_{}_'.format(supertype) in key or '_{}H'.format(supertype) in key:
            if method+"_" in key or method+'i' in key:
                filtered_dict[key] = value
    return filtered_dict

def extract_first_values(filtered_dict):
    """
    从筛选后的字典中提取元组的第一个值。

    Args:
        filtered_dict (dict): 筛选后的数据字典。

    Returns:
        list: 第一个值的列表，排除 NaN。
    """
    first_values = []
    for key, value in filtered_dict.items():
        if len(value) > 0:
            first_val = value[0]
            if not np.isnan(first_val):
                first_values.append(first_val)
            else:
                print(f"NaN found in first value for key: {key}")
        else:
            print(f"No values in tuple for key: {key}")
    return first_values
 
  
    

def length_8_11(folder_path):
    data = {
        8: {'ann': [93.41, 100, 49.38,90.60,98.70, 57, 1], 'consensus': [93.37, 100, 50, 90.20,98.60,57, 1], 'smm': [80.95, 100, 22.22, 73.90,89.30,26, 1],'smmpmbec':[80.55,100,40.63,74,94.40,26,2],'netmhcpan_el':[96.35,100,50,96.20,100,90,1],'netmhcpan_H':[95.97,100,50,94.90,100,90,1],'netmhcstabpan':[88.05,100,25,88.60,99.20,90,1],'pickpocket':[92.29,100,37.50,90.20,99.60,90,3],'ACME':[92.62,100,25,91.96,99.88,72,1],'Anthem':[97.60,100,50,89.80,98.90,84,4],'HLAB':[93.64,100,35,89.59,96.38,90,1],'TransPHLA':[97.70,100,50,96,99.7,90,1],'MGHLA':[98.56,100,0,97.13,100,90,1]},
        9: {'ann': [95.88, 100, 86.45,96.10,98.30, 73, 0], 'consensus': [94.97, 99.35, 50,93.80,98.10, 74, 0], 'smm': [91.81, 99.25, 39.81,91.30,97.80, 74, 1],'smmpmbec':[92.08,99.12,31.07,91.10,97.90,76,3],'netmhcpan_el':[97.62,100,83.71,97.50,99.20,112,0],'netmhcpan_H':[97.04,100,86.55,97.50,99.00,112,0],'netmhcstabpan':[92.58,100,79.19,94.40,97.80,112,0],'pickpocket':[93.99,100,67.13,94.10,98.20,112,0],'ACME':[96.61,100,77.58,96.88,98.73,90,0],'Anthem':[98.24,100,91.57,97.50,99.10,111,0],'HLAB':[94.55,100,84.12,92.92,96.44,112,0],'TransPHLA':[98.96,100,92.39,97.6,99,112,0],'MGHLA':[98.95,100,95.06,98.79,99.59,112,0]},
        10:{'ann':[92.79,100,80.80,93.80,98.40,58,0],'consensus':[94.35,100,81.82,92.50,98.30,58,0],'smm':[92.52,99.52,67.89,88.50,97.50,37,0],'smmpmbec':[93.03,99.48,65.41,90.70,97.80,37,0],'netmhcpan_el':[96.13,100,56.25,95.70,99.30,90,0],'netmhcpan_H':[95.54,100,50,95.70,99.10,90,1],'netmhcstabpan':[92.87,100,59.38,93.00,98.10,90,0],'pickpocket':[93.72,100,62.50,92.80,98.80,90,0],'ACME':[94.81,100,43.75,95.18,98.76,72,1],'Anthem':[97.45,100,81.82,96.2,98.9,84,0],'HLAB':[93.01,100,50,91.78,96.49,90,0],'TransPHLA':[97.47,100,43.75,96.2,99.1,90,1],'MGHLA':[98.45,100,43.75,97.83,99.63,90,1]},
        11:{'ann':[91.85,100,70.32,92.2,97.9,57,0],'consensus':[91.14,100,70.48,89.8,97.5,57,0],'smm':[80.35,95.53,61.82,77.4,90.1,26,0],'smmpmbec':[81.66,96.2,69.39,83.4,93.6,26,0],'netmhcpan_el':[95.17,100,76.19,95.7,99.5,91,0],'netmhcpan_H':[94.31,100,71.73,94.8,99.4,91,0],'netmhcstabpan':[92.13,100,68.14,91.7,98.9,91,0],'pickpocket':[92.99,100,66.33,92.3,99,91,0],'ACME':[93.53,100,72.39,93.83,99,73,0],'Anthem':[96.75,100,55.1,93.3,98.8,78,0],'HLAB':[92.79,100,76.67,90.63,96.83,91,0],'TransPHLA':[97.07,100,80.61,95.7,99.2,91,0],'MGHLA':[98.20,100,66.67,97.7,99.71,91,0]}
        
    }
    # 创建色板
    all_methods = {method for methods in data.values() for method in methods}
    color_palette = sns.color_palette("husl", n_colors=len(all_methods))
    method_colors = {method: color for method, color in zip(all_methods, color_palette)}
    file_list=os.listdir(folder_path)
    

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 6))

    # 设置绘图参数
    width = 0.15  # 箱体宽度，更宽
    gap = 0.013  # 方法之间没有间隙
    group_gap = 0.5  # 长度组之间的间隙，清晰分界

    # 绘制数据
    positions = []  # 记录各个箱体的位置
    already_labeled = False
    for i, (length, methods) in enumerate(sorted(data.items())):
        num_methods = len(methods)
        for j, (method, values) in enumerate(methods.items()):
            print(length,method)
            file_name='{}_len_metrics_max_min.txt'.format(method)
            if file_name in file_list:
                file_path=os.path.join(folder_path,file_name)
            else:
                file_path=os.path.join(folder_path,'IEDB_method_len_metrics_max_min.txt')
            total_perf, max_perf, min_perf, q1_perf, q3_perf, _, _ = values
            if total_perf!=0:
                pos = i * (num_methods * width + group_gap) + j * (width+gap)
                IQR=q3_perf - q1_perf
                lower_bound=q1_perf-1.5*IQR
                upper_bound=q3_perf+1.5*IQR
                if upper_bound>100:
                    upper_bound=100
                print(lower_bound,upper_bound)
                ax.bar(pos, q3_perf - q1_perf, bottom=q1_perf, width=width, color=method_colors[method], edgecolor='black', linewidth=0.1,label=method if i == 0 else "",zorder=3)
                ax.plot([pos, pos], [lower_bound, q1_perf], color='black', linestyle='-', linewidth=0.1,zorder=3)  # 下触须
                ax.plot([pos, pos], [q3_perf, upper_bound], color='black', linestyle='-', linewidth=0.1,zorder=3)  # 上触须
                # 触须上的横线
                ax.plot([pos - width/2, pos + width/2], [lower_bound, lower_bound], color='black', linewidth=0.1,zorder=3)  # 触须下端横线
                ax.plot([pos - width/2, pos + width/2], [upper_bound,upper_bound], color='black', linewidth=0.1,zorder=3)  # 触须上端横线
                ax.plot([pos - width/2+0.015, pos + width/2-0.015], [total_perf, total_perf], color='#505050', linewidth=1.5,zorder=3)  # 总性能线
                positions.append(pos)
                
                data_dict=read_and_parse_file(file_path)  
                filtered_dict=filter_keys_by_length(data_dict,method, length)  
                first_values=extract_first_values(filtered_dict)
                #first_values=first_values*100
                # 找出离群点
                print(lower_bound,upper_bound)
                outliers=[]
                for k in range(len(first_values)):
                    if (float(first_values[k])*100 < lower_bound) or (float(first_values[k])*100 > upper_bound):
                        print(f'{float(first_values[k])*100}不在{lower_bound}和{upper_bound}之间')
                        outliers.append(float(first_values[k])*100)
                    else:
                        print(str(first_values[k])+'不是离群点')
                if len(outliers)==len(first_values):
                    print(method,length,'Error')

                if outliers:
                    # 创建与离群点数量相同的x位置列表
                    x_outliers = [pos] * len(outliers)
                    ax.scatter(x_outliers, outliers, color='black', marker='o', label='Outliers' if not already_labeled else "",s=5, zorder=0.005)
                    already_labeled = True  # 只添加一次 'Outliers' 标签
                else:
                    print(method,length,'无离群点')
    # 设置x轴标签位置和标签
    ax.set_xticks([np.mean(positions[i:i+num_methods]) for i in range(0, len(positions), num_methods)])
    print([np.mean(positions[i:i+num_methods]) for i in range(0, len(positions), num_methods)])
    print(positions)
    ax.set_xticklabels(sorted( data.keys()))
    ax.set_xlim(-width, max(positions) + width)

    # 设置y轴范围
    ax.set_ylim(20, 100)
    ax.set_xlabel('Peptide Length',fontsize=18)
    ax.set_ylabel('AUC',fontsize=18)
    plt.title('length 8-11 Metrics', fontsize=20)

    # 添加图例
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Compression Mark at 20')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
   
    # 添加网格
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, axis='both',color='lightgray', zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    plt.savefig('length 8-11 Metrics.png')
    plt.savefig('length 8-11 Metrics.pdf')
    plt.savefig('length 8-11 Metrics.svg')

    
def length_12_14(folder_path):
    data = {
        12:{'ann':[84.56,100,47.05,84,98.3,56,1],'consensus':[84.99,100,47.12,84.1,98.3,56,1],'smm': [0, 0, 0, 0,0,0, 0],'smmpmbec':[0,0,0,0,0,0,0],'netmhcpan_el':[89,100,0,93.4,100,88,2],'netmhcpan_ba':[87.84,100,0,89.5,100,88,2],'netmhcstabpan':[85.36,100,0,86.2,99.9,88,3],'pickpocket':[0,0,0,0,0,0,0],'ACME':[85.80,100,0,90.70,100,70,1],'Anthem':[92.88,100,50,83.3,97,77,9],'HLAB':[88.31,100,50,85.11,95.83,88,1],'TransPHLA':[93.99,100,72.22,92.9,100,88,1],'MGHLA':[96.26,100,0,96.48,100,88,3]},
        13:{'ann':[77.3,100,0,80.7,100,53,2],'consensus':[78,100,0,80.9,100,53,2],'smm': [0, 0, 0, 0,0,0, 0],'smmpmbec':[0,0,0,0,0,0,0],'netmhcpan_el':[82.46,100,0,88.8,100,76,3],'netmhcpan_ba':[81.13,100,0,86.1,100,76,7],'netmhcstabpan':[79.1,100,0,78.3,100,76,7],'pickpocket':[0,0,0,0,0,0,0],'ACME':[79.45,100,0,88.89,100,61,2],'Anthem':[89.3,100,37.5,75,95.8,63,6],'HLAB':[85.24,100,50,81.25,100,76,1],'TransPHLA':[91.42,100,25,92.3,100,76,3],'MGHLA':[94.56,100,75,95.24,100,76,1]},
        14:{'ann':[64.21,100,0,64.9,100,47,9],'consensus':[64.46,100,0,64.8,100,47,9],'smm': [0, 0, 0, 0,0,0, 0],'smmpmbec':[0,0,0,0,0,0,0],'netmhcpan_el':[70.06,100,0,70.5,100,71,8],'netmhcpan_ba':[67.42,100,0,72,100,71,12],'netmhcstabpan':[66.66,100,0,66.9,100,71,11],'pickpocket':[0,0,0,0,0,0,0],'ACME':[66.52,100,0,75,100,56,6],'Anthem':[82.92,100,50,50,84.8,52,17],'HLAB':[78.9,100,40,68.95,100,71,11],'TransPHLA':[86.38,100,0,85.6,100,71,4],'MGHLA':[90.24,100,0,91.52,100,71,4]}
        
    }
    # 创建色板
    all_methods = {method for methods in data.values() for method in methods}
    color_palette = sns.color_palette("husl", n_colors=len(all_methods))
    method_colors = {method: color for method, color in zip(all_methods, color_palette)}
    file_list=os.listdir(folder_path)

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 6))

    # 设置绘图参数
    width = 0.13  # 箱体宽度，更宽
    gap = 0.02  # 方法之间没有间隙
    group_gap = 0.5  # 长度组之间的间隙，清晰分界

    # 绘制数据
    positions = []  # 记录各个箱体的位置
    already_labeled = True
    for i, (length, methods) in enumerate(sorted(data.items())):
        num_methods = len(methods)
        num=0
        for j, (method, values) in enumerate(methods.items()):
            print(length,method)
            file_name='{}_len_metrics_max_min.txt'.format(method)
            if file_name in file_list:
                file_path=os.path.join(folder_path,file_name)
            else:
                file_path=os.path.join(folder_path,'IEDB_method_len_metrics_max_min.txt')
            total_perf, max_perf, min_perf, q1_perf, q3_perf, _, _ = values
            if total_perf!=0:
                pos = i * (num_methods * width + group_gap) + (j-num) * (width+gap)
                IQR=q3_perf - q1_perf
                lower_bound=q1_perf-1.5*IQR
                upper_bound=q3_perf+1.5*IQR
                if upper_bound>100:
                    upper_bound=100
                ax.bar(pos, q3_perf - q1_perf, bottom=q1_perf, width=width, color=method_colors[method], edgecolor='black', linewidth=0.1,label=method if i == 0 else "",zorder=3)
                ax.plot([pos, pos], [lower_bound, q1_perf], color='black', linestyle='-', linewidth=0.1,zorder=3)  # 下触须
                ax.plot([pos, pos], [q3_perf, upper_bound], color='black', linestyle='-', linewidth=0.1,zorder=3)  # 上触须
                # 触须上的横线
                ax.plot([pos - width/2, pos + width/2], [lower_bound, lower_bound], color='black', linewidth=0.1,zorder=3)  # 触须下端横线
                ax.plot([pos - width/2, pos + width/2], [upper_bound,upper_bound], color='black', linewidth=0.1,zorder=3)  # 触须上端横线
                ax.plot([pos - width/2+0.015, pos + width/2-0.015], [total_perf, total_perf], color='#505050', linewidth=1.5,zorder=3)  # 总性能线
                positions.append(pos)
                data_dict=read_and_parse_file(file_path)  
                filtered_dict=filter_keys_by_length(data_dict,method, length)  
                first_values=extract_first_values(filtered_dict)
                # 找出离群点
                print(lower_bound,upper_bound)
                outliers=[]
                for k in range(len(first_values)):
                    if (float(first_values[k])*100 < lower_bound) or (float(first_values[k])*100 > upper_bound):
                        print(f'{float(first_values[k])*100}不在{lower_bound}和{upper_bound}之间')
                        outliers.append(float(first_values[k])*100)
                    else:
                        print(str(first_values[k])+'不是离群点')
                if len(outliers)==len(first_values):
                    print(method,length,'Error')

                if outliers:
                    # 创建与离群点数量相同的x位置列表
                    x_outliers = [pos] * len(outliers)
                    ax.scatter(x_outliers, outliers, color='black', marker='o', label='Outliers' if not already_labeled else "",s=5, zorder=0.005)
                    already_labeled = True  # 只添加一次 'Outliers' 标签
                else:
                    print(method,length,'无离群点')
            else:
                num=num+1
            
           

    # 设置x轴标签位置和标签
    ax.set_xticks([np.mean(positions[i:i+num_methods]) for i in range(0, len(positions), num_methods)])
    ax.set_xticklabels(sorted( data.keys()))
    ax.set_xlim(-width, max(positions) + width)

    # 设置y轴范围
    ax.set_ylim(0, 100)
    ax.set_xlabel('Peptide Length',fontsize=18)
    ax.set_ylabel('AUC',fontsize=18)
    
    plt.title('length 12-14 Metrics', fontsize=20)

    # 添加图例
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # 添加网格
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, axis='both',color='lightgray', zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    plt.savefig('length 12-14 Metrics.png')
    plt.savefig('length 12-14 Metrics.pdf')
    plt.savefig('length 12-14 Metrics.svg')


def supertype_graph_A(folder_path):
    data={
        'A01':{'ann':[0.9648,1.0,0.923,0.959,0.979,10,0],'consensus':[0.9626,0.985,0.543,0.938,0.97,10,0],'smm':[0.9505,0.978,0.398,0.921,0.969,10,1],'smmpmbec':[0.9511,0.981,0.498,0.927,0.975,10,1],'netmhcpan_el':[0.9808,0.987,0.904,0.968,0.982,10,0],'netmhcpan_ba':[0.9781,1.0,0.951,0.963,0.983,10,0],'netmhcstabpan':[0.9672,0.98,0.905,0.939,0.975,10,0],'pickpocket':[0.9531,1.0,0.906,0.936,0.965,10,0],'ACME':[0.9714,0.991,0.917,0.968,0.984,9,0],'Anthem':[0.9828,0.998,0.931,0.967,0.985,10,0],'HLAB':[0.951,0.9643,0.8556,0.9165,0.9567,10,0],'TransPHLA':[0.9831,0.996,0.939,0.964,0.987,10,0],'MGHLA':[0.9893,1.0,0.9545,0.9829,0.9919,10,0]},
        'A02':{'ann':[0.9399,1.0,0.918,0.949,0.986,12,0],'consensus':[0.9381,0.986,0.872,0.944,0.961,12,0],'smm':[0.932,0.982,0.673,0.935,0.96,12,0],'smmpmbec':[0.9356,0.985,0.751,0.934,0.965,12,0],'netmhcpan_el':[0.9479,0.995,0.912,0.953,0.98,16,0],'netmhcpan_ba':[0.946,1.0,0.921,0.961,0.981,16,0],'netmhcstabpan':[0.939,1.0,0.921,0.952,0.973,16,0],'pickpocket':[0.9401,1.0,0.921,0.945,0.968,16,0],'ACME':[0.9466,0.996,0.927,0.957,0.977,15,0],'Anthem':[0.9626,0.999,0.936,0.955,0.978,16,0],'HLAB':[0.911,0.9722,0.8886,0.9107,0.9431,16,0],'TransPHLA':[0.9628,1.0,0.95,0.967,0.984,16,0],'MGHLA':[0.9772,1,0.9667,0.9811,0.9903,16,0]},
        'A03':{'ann':[0.9678,1.0,0.702,0.972,0.98,7,0],'consensus':[0.9686,0.979,0.792,0.898,0.975,7,0],'smm':[0.9639,0.974,0.715,0.921,0.97,7,0],'smmpmbec':[0.9661,0.977,0.761,0.95,0.97,7,0],'netmhcpan_el':[0.9745,0.983,0.969,0.971,0.98,7,0],'netmhcpan_ba':[0.976,0.982,0.969,0.971,0.978,7,0],'netmhcstabpan':[0.9671,0.973,0.941,0.961,0.973,7,0],'pickpocket':[0.9637,0.975,0.938,0.962,0.968,7,0],'ACME':[0.9744,0.981,0.896,0.972,0.978,7,0],'Anthem':[0.9796,0.986,0.975,0.978,0.982,7,0],'HLAB':[0.942,0.9468,0.8412,0.9389,0.9456,7,0],'TransPHLA':[0.9793,0.985,0.924,0.976,0.981,7,0],'MGHLA':[0.9869,1.0,0.983,0.9867,0.9915,7,0]},
        'A24':{'ann':[0.972,0.973,0.968,0.97,0.972,3,0],'consensus':[0.9685,0.97,0.966,0.966,0.968,3,0],'smm':[0.9639,0.966,0.961,0.963,0.965,3,0],'smmpmbec':[0.9677,0.969,0.962,0.964,0.968,3,0],'netmhcpan_el':[0.9758,0.978,0.953,0.96,0.976,5,0],'netmhcpan_ba':[0.9755,0.978,0.921,0.961,0.977,5,0],'netmhcstabpan':[0.9738,0.977,0.93,0.958,0.975,5,0],'pickpocket':[0.9744,0.976,0.926,0.961,0.975,5,0],'ACME':[0.9697,0.982,0.969,0.97,0.977,3,0],'Anthem':[0.9784,0.986,0.946,0.947,0.977,5,0],'HLAB':[0.942,0.9473,0.8956,0.9205,0.9465,5,0],'TransPHLA':[0.9796,0.987,0.94,0.977,0.983,5,0],'MGHLA':[0.9865,0.9928,0.9615,0.9832,0.9882,5,0]},
        
        }
    
    # 创建色板
    all_methods = {method for methods in data.values() for method in methods}
    color_palette = sns.color_palette("husl", n_colors=len(all_methods))
    method_colors = {method: color for method, color in zip(all_methods, color_palette)}
    file_list=os.listdir(folder_path)

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 6))
    #fig, ax_main = plt.subplots(figsize=(10, 6))

    # 设置绘图参数
    width = 0.15  # 箱体宽度，更宽
    gap = 0.013  # 方法之间没有间隙
    group_gap = 0.5  # 长度组之间的间隙，清晰分界

    # 绘制数据
    positions = []  # 记录各个箱体的位置
    already_labeled = True
    for i, (supertype, methods) in enumerate(sorted(data.items())):
        num_methods = len(methods)
        for j, (method, values) in enumerate(methods.items()):
            print(supertype,method)
            file_name='{}_supertypes_hla_metrics.txt'.format(method)
            if file_name in file_list:
                file_path=os.path.join(folder_path,file_name)
            else:
                file_path=os.path.join(folder_path,'IEDB_method_supertypes_hla_metrics.txt')
            total_perf, max_perf, min_perf, q1_perf, q3_perf, _, _ = values
            total_perf, max_perf, min_perf, q1_perf, q3_perf=total_perf*100, max_perf*100, min_perf*100, q1_perf*100, q3_perf*100
            pos = i * (num_methods * width + group_gap) + j * (width+gap)
            IQR=q3_perf - q1_perf
            lower_bound=q1_perf-1.5*IQR
            upper_bound=q3_perf+1.5*IQR
            if upper_bound>100:
                upper_bound=100
            ax.bar(pos, q3_perf - q1_perf, bottom=q1_perf, width=width, color=method_colors[method], edgecolor='black', linewidth=0.1,label=method if i == 0 else "",zorder=3)
            ax.plot([pos, pos], [lower_bound, q1_perf], color='black', linestyle='-', linewidth=0.1,zorder=3)  # 下触须
            ax.plot([pos, pos], [q3_perf, upper_bound], color='black', linestyle='-', linewidth=0.1,zorder=3)  # 上触须
            # 触须上的横线
            ax.plot([pos - width/2, pos + width/2], [lower_bound, lower_bound], color='black', linewidth=0.1,zorder=3)  # 触须下端横线
            ax.plot([pos - width/2, pos + width/2], [upper_bound,upper_bound], color='black', linewidth=0.1,zorder=3)  # 触须上端横线
            ax.plot([pos - width/2+0.015, pos + width/2-0.015], [total_perf, total_perf], color='#505050', linewidth=1.5,zorder=3)  # 总性能线
            positions.append(pos)
            
            data_dict=read_and_parse_file(file_path)  
            filtered_dict=filter_keys_by_supertype(data_dict,method, supertype)
            first_values=extract_first_values(filtered_dict)
            # 找出离群点
            print(lower_bound,upper_bound)
            outliers=[]
            for k in range(len(first_values)):
                if (float(first_values[k])*100 < lower_bound) or (float(first_values[k])*100 > upper_bound):
                    print(f'{float(first_values[k])*100}不在{lower_bound}和{upper_bound}之间')
                    outliers.append(float(first_values[k])*100)
                else:
                    print(str(first_values[k])+'不是离群点')
            if len(outliers)==len(first_values):
                print(method,supertype,'Error')

            if outliers:
                #print(method,length,outliers)
                # 创建与离群点数量相同的x位置列表
                x_outliers = [pos] * len(outliers)
                ax.scatter(x_outliers, outliers, color='black', marker='o', label='Outliers' if not already_labeled else "",s=5, zorder=0.005)
                already_labeled = True  # 只添加一次 'Outliers' 标签
            else:
                print(method,supertype,'无离群点')
            
    
    ax.set_ylim(40, 100)  # 主要展示的数据范围
    ax.set_xticks([np.mean(positions[i:i+num_methods]) for i in range(0, len(positions), num_methods)])
    ax.xaxis.set_major_locator(FixedLocator([np.mean(positions[i:i+num_methods]) for i in range(0, len(positions), num_methods)]))
    ax.set_ylabel('AUC',fontsize=18)
    
    ax.set_xlim(-width, max(positions) + width)
    ax.set_xticklabels(sorted( data.keys()))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, axis='both',color='lightgray', zorder=0)
    # 添加压缩区域的象征线
    ax.axhline(y=40, color='red', linestyle='--', linewidth=2, label='Compression Mark at 40')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('supertypeA', fontsize=20)

    # 绘制折线或数据示例（可自定义添加具体数据绘制代码）
    plt.tight_layout()
    plt.show()
    
    plt.savefig('supertypeA.png')
    plt.savefig('supertypeA.pdf')
    plt.savefig('supertypeA.svg')
    
    

    
def supertype_graph_B(folder_path):
    data={
        
        'B07':{'ann':[0.9464,0.981,0.91,0.945,0.975,7,0],'consensus':[0.9508,0.977,0.679,0.915,0.97,7,0],'smm':[0.935,0.968,0.587,0.892,0.956,7,0],'smmpmbec':[0.9399,0.97,0.692,0.901,0.965,7,0],'netmhcpan_el':[0.9719,0.995,0.912,0.954,0.989,10,0],'netmhcpan_ba':[0.9676,0.993,0.939,0.958,0.989,10,0],'netmhcstabpan':[0.9572,0.99,0.868,0.947,0.985,10,0],'pickpocket':[0.9588,0.989,0.83,0.944,0.982,10,0],'ACME':[0.9599,0.99,0.907,0.958,0.983,9,0],'Anthem':[0.9798,0.995,0.963,0.969,0.984,10,0],'HLAB':[0.947,0.9716,0.8646,0.9356,0.9655,10,0],'TransPHLA':[0.9825,0.995,0.966,0.974,0.991,10,0],'MGHLA':[0.9897,1,0.983,0.9857,0.9975,10,0]},
        'B27':{'ann':[0.8347,1.0,0.784,0.958,0.991,9,0],'consensus':[0.8307,0.992,0.742,0.865,0.98,9,0],'smm':[0.8732,0.993,0.673,0.836,0.979,9,0],'smmpmbec':[0.8681,0.991,0.311,0.763,0.974,11,2],'netmhcpan_el':[0.8939,0.997,0.778,0.981,0.995,23,0],'netmhcpan_ba':[0.9047,1.0,0.758,0.979,0.991,23,0],'netmhcstabpan':[0.8772,0.99,0.762,0.959,0.981,23,0],'pickpocket':[0.9043,0.992,0.671,0.939,0.985,23,0],'ACME':[0.8733,1.0,0.747,0.971,0.987,22,0],'Anthem':[0.9637,1.0,0.877,0.984,0.994,23,0],'HLAB':[0.908,0.9835,0.8233,0.925,0.973,23,0],'TransPHLA':[0.9663,0.999,0.889,0.979,0.994,23,0],'MGHLA':[0.9781,1,0.918,0.9901,0.9975,23,0]},
        'B44':{'ann':[0.9829,0.99,0.977,0.979,0.985,6,0],'consensus':[0.9833,0.988,0.974,0.981,0.986,6,0],'smm':[0.9774,0.985,0.964,0.973,0.982,6,0],'smmpmbec':[0.9754,0.985,0.964,0.97,0.982,6,0],'netmhcpan_el':[0.9879,0.999,0.983,0.988,0.994,13,0],'netmhcpan_ba':[0.9879,0.998,0.984,0.986,0.992,13,0],'netmhcstabpan':[0.97,0.988,0.944,0.965,0.98,13,0],'pickpocket':[0.9781,1.0,0.948,0.975,0.987,13,0],'ACME':[0.9793,0.999,0.876,0.979,0.989,13,0],'Anthem':[0.9899,0.994,0.947,0.986,0.992,13,0],'HLAB':[0.968,0.9757,0.9463,0.9632,0.9729,13,0],'TransPHLA':[0.9906,0.998,0.984,0.988,0.994,13,0],'MGHLA':[0.9949,0.9994,0.9914,0.9941,0.9966,13,0]},
        'B58':{'ann':[0.9445,0.966,0.935,0.95,0.965,3,0],'consensus':[0.9299,0.971,0.915,0.939,0.966,3,0],'smm':[0.9222,0.969,0.9,0.932,0.966,3,0],'smmpmbec':[0.9365,0.967,0.923,0.944,0.966,3,0],'netmhcpan_el':[0.9607,0.978,0.95,0.966,0.973,4,0],'netmhcpan_ba':[0.9573,0.977,0.946,0.962,0.975,4,0],'netmhcstabpan':[0.9547,0.978,0.943,0.961,0.972,4,0],'pickpocket':[0.9535,0.971,0.947,0.949,0.97,4,0],'ACME':[0.9519,0.978,0.945,0.956,0.976,4,0],'Anthem':[0.9722,0.997,0.968,0.973,0.977,4,0],'HLAB':[0.933,0.9444,0.9258,0.9324,0.9424,4,0],'TransPHLA':[0.9756,0.981,0.971,0.974,0.981,4,0],'MGHLA':[0.9847,0.9875,0.9825,0.9848,0.9872,4,0]},
        'B62':{'ann':[0.9478,0.974,0.908,0.924,0.957,3,0],'consensus':[0.9568,0.975,0.505,0.813,0.952,4,0],'smm':[0.938,0.975,0.504,0.794,0.97,4,0],'smmpmbec':[0.9486,0.977,0.531,0.818,0.974,4,0],'netmhcpan_el':[0.9737,0.979,0.837,0.955,0.975,5,0],'netmhcpan_ba':[0.9637,0.979,0.904,0.959,0.969,5,0],'netmhcstabpan':[0.9601,0.98,0.841,0.957,0.959,5,0],'pickpocket':[0.9579,0.971,0.763,0.932,0.945,5,0],'ACME':[0.9559,0.978,0.776,0.921,0.966,5,0],'Anthem':[0.9817,0.983,0.97,0.976,0.983,5,0],'HLAB':[0.945,0.9507,0.8704,0.929,0.9419,5,0],'TransPHLA':[0.9795,0.982,0.965,0.971,0.974,5,0],'MGHLA':[0.9873,0.995,0.9832,0.9838,0.9884,5,0]},
        
        }
    
     # 创建色板
    all_methods = {method for methods in data.values() for method in methods}
    color_palette=sns.color_palette("husl",n_colors=len(all_methods))
    method_colors = {method: color for method, color in zip(all_methods, color_palette)}
    file_list=os.listdir(folder_path)

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 6))

    # 设置绘图参数
    width = 0.13  # 箱体宽度，更宽
    gap = 0.02  # 方法之间没有间隙
    group_gap = 0.5  # 长度组之间的间隙，清晰分界

    # 绘制数据
    positions = []  # 记录各个箱体的位置
    already_labeled = True
    for i, (supertype, methods) in enumerate(sorted(data.items())):
        num_methods = len(methods)
        for j, (method, values) in enumerate(methods.items()):
            print(supertype,method)
            file_name='{}_supertypes_hla_metrics.txt'.format(method)
            if file_name in file_list:
                file_path=os.path.join(folder_path,file_name)
            else:
                file_path=os.path.join(folder_path,'IEDB_method_supertypes_hla_metrics.txt')
            total_perf, max_perf, min_perf, q1_perf, q3_perf, _, _ = values
            total_perf, max_perf, min_perf, q1_perf, q3_perf=total_perf*100, max_perf*100, min_perf*100, q1_perf*100, q3_perf*100
            pos = i * (num_methods * width + group_gap) + j * (width+gap)
            IQR=q3_perf - q1_perf
            lower_bound=q1_perf-1.5*IQR
            upper_bound=q3_perf+1.5*IQR
            if upper_bound>100:
                upper_bound=100
            ax.bar(pos, q3_perf - q1_perf, bottom=q1_perf, width=width, color=method_colors[method], edgecolor='black', linewidth=0.1,label=method if i == 0 else "",zorder=3)
            ax.plot([pos, pos], [lower_bound, q1_perf], color='black', linestyle='-', linewidth=0.1,zorder=3)  # 下触须
            ax.plot([pos, pos], [q3_perf, upper_bound], color='black', linestyle='-', linewidth=0.1,zorder=3)  # 上触须
            # 触须上的横线
            ax.plot([pos - width/2, pos + width/2], [lower_bound, lower_bound], color='black', linewidth=0.1,zorder=3)  # 触须下端横线
            ax.plot([pos - width/2, pos + width/2], [upper_bound,upper_bound], color='black', linewidth=0.1,zorder=3)  # 触须上端横线
            ax.plot([pos - width/2+0.015, pos + width/2-0.015], [total_perf, total_perf], color='#505050', linewidth=1.5,zorder=3)  # 总性能线
            positions.append(pos)
            
            data_dict=read_and_parse_file(file_path)  
            filtered_dict=filter_keys_by_supertype(data_dict,method, supertype)
            first_values=extract_first_values(filtered_dict)
            # 找出离群点
            print(lower_bound,upper_bound)
            outliers=[]
            for k in range(len(first_values)):
                if (float(first_values[k])*100 < lower_bound) or (float(first_values[k])*100 > upper_bound):
                    print(f'{float(first_values[k])*100}不在{lower_bound}和{upper_bound}之间')
                    outliers.append(float(first_values[k])*100)
                else:
                    print(str(first_values[k])+'不是离群点')
            if len(outliers)==len(first_values):
                print(method,supertype,'Error')

            if outliers:
                x_outliers = [pos] * len(outliers)
                ax.scatter(x_outliers, outliers, color='black', marker='o', label='Outliers' if not already_labeled else "",s=5, zorder=0.005)
                already_labeled = True  # 只添加一次 'Outliers' 标签
            else:
                print(method,supertype,'无离群点')
            
           

    ax.set_ylim(20, 100)  # 主要展示的数据范围
    ax.set_xticks([np.mean(positions[i:i+num_methods]) for i in range(0, len(positions), num_methods)])
    ax.xaxis.set_major_locator(FixedLocator([np.mean(positions[i:i+num_methods]) for i in range(0, len(positions), num_methods)]))
    ax.set_ylabel('AUC',fontsize=18)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, axis='both',color='lightgray', zorder=0)
    ax.set_xlim(-width, max(positions) + width)
    ax.set_xticklabels(sorted( data.keys()))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.title('supertypeB', fontsize=20)
    
    # 添加压缩区域的象征线
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Compression Mark at 20')
    # 绘制折线或数据示例（可自定义添加具体数据绘制代码）
    plt.tight_layout()
    plt.show()
    
    plt.savefig('supertypeB.png')
    plt.savefig('supertypeB.pdf')
    plt.savefig('supertypeB.svg')
    
def supertype_graph_C(ax,folder_path):
    data={
        
        'C01':{'ann':[0.8888,0.945,0.779,0.928,0.943,5,0],'consensus':[0.8632,0.942,0.777,0.839,0.935,5,0],'smm':[0.7065,0.922,0.724,0.749,0.913,5,0],'smmpmbec':[0.7197,0.924,0.756,0.798,0.906,5,0],'netmhcpan_el':[0.963,0.986,0.929,0.963,0.971,8,0],'netmhcpan_ba':[0.9568,0.983,0.929,0.961,0.973,8,0],'netmhcstabpan':[0.8691,0.943,0.837,0.862,0.915,8,0],'pickpocket':[0.9032,0.985,0.84,0.902,0.94,8,0],'ACME':[0,0,0,0,0,0,0],'Anthem':[0.9786,0.989,0.952,0.973,0.986,8,0],'HLAB':[0.942,0.963,0.9123,0.9313,0.9562,8,0],'TransPHLA':[0.9795,0.991,0.958,0.979,0.986,8,0],'MGHLA':[0.9886,0.9953,0.9743,0.9861,0.9927,8,0]},
        'C02':{'ann':[0.9187,0.957,0.9,0.924,0.952,3,0],'consensus':[0.8952,0.96,0.838,0.863,0.925,3,0],'smm':[0.747,0.932,0.678,0.786,0.914,3,0],'smmpmbec':[0.7488,0.944,0.677,0.798,0.923,3,0],'netmhcpan_el':[0.9612,0.995,0.921,0.968,0.978,5,0],'netmhcpan_ba':[0.9538,0.995,0.925,0.966,0.974,5,0],'netmhcstabpan':[0.8635,0.943,0.77,0.88,0.93,5,0],'pickpocket':[0.8817,0.928,0.811,0.89,0.911,5,0],'ACME':[0,0,0,0,0,0,0],'Anthem':[0.9781,0.993,0.962,0.976,0.982,5,0],'HLAB':[0.945,0.9763,0.9286,0.944,0.9502,5,0],'TransPHLA':[0.9811,0.996,0.972,0.98,0.983,5,0],'MGHLA':[0.9897,0.9967,0.9837,0.9896,0.9909,5,0]},
        'C07':{'ann':[0.9339,0.945,0.923,0.928,0.939,2,0],'consensus':[0.8865,0.904,0.871,0.88,0.896,2,0],'smm':[0.7653,0.826,0.75,0.769,0.807,2,0],'smmpmbec':[0.7761,0.812,0.761,0.774,0.799,2,0],'netmhcpan_el':[0.9603,0.965,0.955,0.959,0.964,3,0],'netmhcpan_ba':[0.9595,0.967,0.961,0.962,0.965,3,0],'netmhcstabpan':[0.8805,0.902,0.869,0.872,0.889,3,0],'pickpocket':[0.8183,0.848,0.8,0.801,0.824,3,0],'ACME':[0,0,0,0,0,0,0],'Anthem':[0.974,0.976,0.973,0.973,0.974,3,0],'HLAB':[0.941,0.943,0.9404,0.9404,0.9418,3,0],'TransPHLA':[0.979,0.981,0.976,0.978,0.981,3,0],'MGHLA':[0.9891,0.9903,0.9885,0.9887,0.9896,3,0]}
        }
    
     # 创建色板
    all_methods = {method for methods in data.values() for method in methods}
    color_palette = sns.color_palette("husl", n_colors=len(all_methods))
    method_colors = {method: color for method, color in zip(all_methods, color_palette)}
    file_list=os.listdir(folder_path)

    # 设置绘图参数
    width = 0.10  # 箱体宽度，更宽
    gap = 0.02  # 方法之间没有间隙
    group_gap = 0.8  # 长度组之间的间隙，清晰分界

    # 绘制数据
    positions = []  # 记录各个箱体的位置
    already_labeled = True
    for i, (supertype, methods) in enumerate(sorted(data.items())):
        num_methods = len(methods)
        
        for j, (method, values) in enumerate(methods.items()):
            print(supertype,method)
            file_name='{}_supertypes_hla_metrics.txt'.format(method)
            if file_name in file_list:
                file_path=os.path.join(folder_path,file_name)
            else:
                file_path=os.path.join(folder_path,'IEDB_method_supertypes_hla_metrics.txt')
            total_perf, max_perf, min_perf, q1_perf, q3_perf, _, _ = values
            total_perf, max_perf, min_perf, q1_perf, q3_perf=total_perf*100, max_perf*100, min_perf*100, q1_perf*100, q3_perf*100
            pos = i * (num_methods * width + group_gap) + j * (width+gap)
            IQR=q3_perf - q1_perf
            lower_bound=q1_perf-1.5*IQR
            upper_bound=q3_perf+1.5*IQR
            if upper_bound>100:
                upper_bound=100
            ax.bar(pos, q3_perf - q1_perf, bottom=q1_perf, width=width, color=method_colors[method], edgecolor='black', linewidth=0.1,label=method if i == 0 else "",zorder=3)
            ax.plot([pos, pos], [lower_bound, q1_perf], color='black', linestyle='-', linewidth=0.1,zorder=3)  # 下触须
            ax.plot([pos, pos], [q3_perf, upper_bound], color='black', linestyle='-', linewidth=0.1,zorder=3)  # 上触须
            # 触须上的横线
            ax.plot([pos - width/2, pos + width/2], [lower_bound, lower_bound], color='black', linewidth=0.1,zorder=3)  # 触须下端横线
            ax.plot([pos - width/2, pos + width/2], [upper_bound,upper_bound], color='black', linewidth=0.1,zorder=3)  # 触须上端横线
            ax.plot([pos - width/2+0.015, pos + width/2-0.015], [total_perf, total_perf], color='#505050', linewidth=1.5,zorder=3)  # 总性能线
            positions.append(pos)
            
            data_dict=read_and_parse_file(file_path)  
            filtered_dict=filter_keys_by_supertype(data_dict,method, supertype)
            first_values=extract_first_values(filtered_dict)
            print('提取数量:',len(first_values))
            # 找出离群点
            print(lower_bound,upper_bound)
            outliers=[]
            for k in range(len(first_values)):
                if (float(first_values[k])*100 < lower_bound) or (float(first_values[k])*100 > upper_bound):
                    print(f'{float(first_values[k])*100}不在{lower_bound}和{upper_bound}之间')
                    outliers.append(float(first_values[k])*100)
                    
                else:
                    print(str(first_values[k])+'不是离群点')
            if len(outliers)==len(first_values):
                print(method,supertype,'Error')

            if outliers:
                print('离群点数量',len(outliers))
                x_outliers = [pos] * len(outliers)
                ax.scatter(x_outliers, outliers, color='black', marker='o', label='Outliers' if not already_labeled else "",s=5, zorder=0.005)
                already_labeled = True  # 只添加一次 'Outliers' 标签
            else:
                print(method,supertype,'无离群点')
            
           

    ax.set_ylim(50, 100)  # 主要展示的数据范围
    ax.set_xticks([np.mean(positions[i:i+num_methods]) for i in range(0, len(positions), num_methods)])
    ax.xaxis.set_major_locator(FixedLocator([np.mean(positions[i:i+num_methods]) for i in range(0, len(positions), num_methods)]))
    ax.set_ylabel('AUC',fontsize=18)
    
    ax.set_xlim(-width, max(positions) + width)
    ax.set_xticklabels(sorted( data.keys()))
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, axis='both',color='lightgray', zorder=0)
    # 添加压缩区域的象征线
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Compression Mark')
    plt.title('supertypeC', fontsize=20)

   
    plt.tight_layout()
    plt.show()
    plt.savefig('supertypeC.png')
    plt.savefig('supertypeC.pdf')
    plt.savefig('supertypeC.svg')
    
    
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

def select_colors_for_methods():
    # 定义13种颜色，这里使用了HEX颜色代码
    color_palette = [
        '#FF5733', '#33FF57', '#3357FF', '#F333FF', 
        '#57FF33', '#FF3357', '#33F5FF', '#FF5733', 
        '#F35733', '#3357F3', '#F33357', '#57F333', 
        '#FF33F5'
    ]
    
    # 定义13种方法
    methods = [
        "Ann", "Consensus", "Smm", "Smmpmbec", 
        "netmhcpan_el", "netmhcpan_ba", "netmhcstabpan", "pickpocket",
        "ACME", "Anthem", "HLAB", "TransPHLA", "MGHLA"
    ]
    
    # 创建方法与颜色的对应字典
    method_color_mapping = dict(zip(methods, color_palette))
    
    # 保存对应关系到JSON文件，以便以后使用
    with open('method_color_mapping.json', 'w') as f:
        json.dump(method_color_mapping, f, indent=4)
    
    return method_color_mapping



if __name__ == '__main__':
    
    folder_path='../baseline_length_supertype'
    length_8_11(folder_path)
    
    length_12_14(folder_path)
    supertype_graph_A(folder_path)
    supertype_graph_B(folder_path)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    supertype_graph_C(ax1,folder_path)
    # 提取图例
    
    handles1, labels1 =ax1.get_legend_handles_labels()
    handles =  handles1
    labels =  labels1
    # 创建一个新的空白图形
    fig_leg = plt.figure(figsize=(4, 1))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.legend(handles, labels, loc='center', ncol=1,frameon=True)
    ax_leg.axis('off')  # 隐藏坐标轴

    # 保存图例
    fig_leg.savefig('legend_2_J_4.pdf', dpi=300, bbox_inches='tight', transparent=True)
    fig_leg.savefig('legend_2_J_4.svg', dpi=300, bbox_inches='tight', transparent=True)
 