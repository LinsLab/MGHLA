import os
import numpy as np
import pandas as pd
from performance import *

def length_metrics(file1,len_pos,score_pos,method,ic50_or_rank):
    data_list=[]
    with open(file1, 'r') as file:
        for line in file:
            line=line.strip()
            
            # 去除每行的换行符并添加到列表
            #data_list.append(line.strip())
            values = line.strip().split(',')

            # 将分隔后的值转换成相应的数据类型，例如整数或浮点数
            values = [eval(value) if value.replace('.', '', 1).isdigit() else value for value in values]

            # 将这些值作为一个列表添加到外层的列表中
            data_list.append(values)
    file.close()
    
    length=set(row[len_pos] for row in data_list)
    len_metrics=dict()
    len_data_dict=dict()
    len_metrics={key: None for key in length}
    #len_data_dict={key: [[] for _ in range(3)] for key in length}
    len_data_dict={key:[] for key in length}
    for i in range(len(data_list)):
        len_data_dict[data_list[i][len_pos]].append(data_list[i])
        
    for key in len_data_dict.keys():
        length_list=len_data_dict[key]
        #print(supertype_data_list)
        y_true=[int(d[-2]) for d in length_list]
        y_preb=[int(d[-1]) for d in length_list]
        y_prob=[float(d[score_pos]) for d in length_list]
    
        metrics_set=performances(y_true,y_preb,y_prob,ic50_or_rank,False)
        len_metrics[key]=metrics_set
    print(len_metrics)
    
    

def length_metrics_max_min(file1,len_pos,score_pos,method,ic50_or_rank):
     
    data_list=[]
    with open(file1, 'r') as file:
        for line in file:
            line=line.strip()
            
            # 去除每行的换行符并添加到列表
            #data_list.append(line.strip())
            values = line.strip().split(',')

            # 将分隔后的值转换成相应的数据类型，例如整数或浮点数
            values = [eval(value) if value.replace('.', '', 1).isdigit() else value for value in values]

            # 将这些值作为一个列表添加到外层的列表中
            data_list.append(values)
    file.close()
    length=set(row[len_pos] for row in data_list)
    len_metrics=dict()
    len_data_dict=dict()
    len_metrics={key: None for key in length}
    #len_data_dict={key: [[] for _ in range(3)] for key in length}
    len_data_dict={key:[] for key in length}
    for i in range(len(data_list)):
        len_data_dict[data_list[i][len_pos]].append(data_list[i])
    
    metrics_dict=dict()    
    
    for key in len_data_dict.keys():
        length_data_list=len_data_dict[key]
        
        hla_name=set(length_data_list[i][0] for i in range(len(length_data_list)))
        length_hla_dict={key:[] for key in hla_name}
        for i in range(len(length_data_list)):
            length_hla_dict[length_data_list[i][0]].append(length_data_list[i])
        
        for key1 in sorted(length_hla_dict.keys()):
            length_hla_list=length_hla_dict[key1]
            #print(supertype_data_list)
            y_true=[int(d[-2]) for d in length_hla_list]
            y_preb=[int(d[-1]) for d in length_hla_list]
            y_prob=[float(d[score_pos]) for d in length_hla_list]
        
            metrics_set=performances(y_true,y_preb,y_prob,ic50_or_rank,False)
            metrics_dict[method+'independent1_'+str(key)+"_"+key1]=metrics_set
    
    
    #file2='/home1/layomi/项目代码/MMGHLA_CT_blousm/results/test_set_metrics/length_new/len_metrics_max_min.txt'
    file2='/home1/layomi/项目代码/MGHLA/baseline_length_supertype/MGHLA/len_metrics_max_min.txt'
    recording_w(file2, metrics_dict,'a+')
    

def supertype_metrics(method,file1,score_pos,ic50_or_rank):
    data_list=[]
    with open(file1, 'r') as file:
        for line in file:
            values = line.strip()
            if line!='':
                # 去除每行的换行符并添加到列表
                #data_list.append(line.strip())
                values = line.strip().split(',')

                # 将分隔后的值转换成相应的数据类型，例如整数或浮点数
                values = [eval(value) if value.replace('.', '', 1).isdigit() else value for value in values]

                # 将这些值作为一个列表添加到外层的列表中
                data_list.append(values)
    file.close()
    
    
    #构建超型对应的HLA字典
    supertype_hla_dict=dict()
    supertype_dict=dict()
    supertype_list=[]
    supertype_file='/home1/layomi/项目代码/MMGHLA_CT/data/hla_hla/supertypes2008.txt'
    with open(supertype_file, 'r') as file:
        for line in file:
            values = line.strip()
            # 去除每行的换行符并添加到列表
            #data_list.append(line.strip())
            if values!='':
                values = line.strip().split()

                # 将分隔后的值转换成相应的数据类型，例如整数或浮点数
                values = [eval(value) if value.replace('.', '', 1).isdigit() else value for value in values]

                # 将这些值作为一个列表添加到外层的列表中
                supertype_list.append(values)
    file.close()
    #print(len(supertype_list))
    #print(len(supertype_list[0]))
    #print(supertype_list)
    supertype_key=set(d[1] for d in supertype_list)
    supertype_dict={k:[] for k in supertype_key}
    for i in range(len(supertype_list)):
        supertype_dict[supertype_list[i][1]].append(supertype_list[i][0])
      
    print(supertype_dict)
    #对应超型字典将测试数据分类到不同超型键对应的值中  
    supertype_hla_dict={k:[] for k in supertype_key}
    for i in range(len(data_list)):
        for k in supertype_dict.keys():
            if data_list[i][0] in supertype_dict[k]:
                supertype_hla_dict[k].append(data_list[i])
                
    metrics_dict=dict()            
    for key in sorted(supertype_hla_dict.keys()):
        supertype_data_list=supertype_hla_dict[key]
        #print(supertype_data_list)
        
        y_true=[int(d[-2]) for d in supertype_data_list]
        y_preb=[int(d[-1]) for d in supertype_data_list]
        y_prob=[float(d[score_pos]) for d in supertype_data_list]
        
        metrics_set=performances(y_true,y_preb,y_prob,ic50_or_rank)
        metrics_dict[method+'independent1'+key]=metrics_set
        
        
    write_file='/home1/layomi/项目代码/MGHLA/baseline_length_supertype/MGHLA/supertypes_metrics.txt'
    recording_w(write_file,metrics_dict,'a+')
    

    
    
def supertype_hla_metrics(method,file1,score_pos,ic50_or_rank):
    data_list=[]
    with open(file1, 'r') as file:
        for line in file:
            values = line.strip()
            if line!='':
                # 去除每行的换行符并添加到列表
                #data_list.append(line.strip())
                values = line.strip().split(',')

                # 将分隔后的值转换成相应的数据类型，例如整数或浮点数
                values = [eval(value) if value.replace('.', '', 1).isdigit() else value for value in values]

                # 将这些值作为一个列表添加到外层的列表中
                data_list.append(values)
    file.close()
    
    
    #构建超型对应的HLA字典
    supertype_hla_dict=dict()
    supertype_dict=dict()
    supertype_list=[]
    supertype_file='/home1/layomi/项目代码/MMGHLA_CT/data/hla_hla/supertypes2008.txt'
    with open(supertype_file, 'r') as file:
        for line in file:
            values = line.strip()
            # 去除每行的换行符并添加到列表
            #data_list.append(line.strip())
            if values!='':
                values = line.strip().split()

                # 将分隔后的值转换成相应的数据类型，例如整数或浮点数
                values = [eval(value) if value.replace('.', '', 1).isdigit() else value for value in values]

                # 将这些值作为一个列表添加到外层的列表中
                supertype_list.append(values)
    file.close()
    #print(len(supertype_list))
    #print(len(supertype_list[0]))
    #print(supertype_list)
    supertype_key=set(d[1] for d in supertype_list)
    supertype_dict={k:[] for k in supertype_key}
    for i in range(len(supertype_list)):
        supertype_dict[supertype_list[i][1]].append(supertype_list[i][0])
      
    print(supertype_dict)
    #对应超型字典将测试数据分类到不同超型键对应的值中  
    supertype_hla_dict={k:[] for k in supertype_key}
    for i in range(len(data_list)):
        for k in supertype_dict.keys():
            if data_list[i][0] in supertype_dict[k]:
                supertype_hla_dict[k].append(data_list[i])
                break
                
    metrics_dict=dict()            
    for key in sorted(supertype_hla_dict.keys()):
        supertype_data_list=supertype_hla_dict[key]
        
        hla_name=set(supertype_data_list[i][0] for i in range(len(supertype_data_list)))
        supertype_hla_key_dict={key0:[] for key0 in hla_name}
        for i in range(len(supertype_data_list)):
            supertype_hla_key_dict[supertype_data_list[i][0]].append(supertype_data_list[i])
            
        for key1 in supertype_hla_key_dict.keys():
            supertype_hla_key_list=supertype_hla_key_dict[key1]
            
            y_true=[int(d[-2]) for d in supertype_hla_key_list]
            y_preb=[int(d[-1]) for d in supertype_hla_key_list]
            y_prob=[float(d[score_pos]) for d in supertype_hla_key_list]
        
            metrics_set=performances(y_true,y_preb,y_prob,ic50_or_rank)
            metrics_dict[method+'independent1_'+str(key)+"_"+key1]=metrics_set
        
        
    write_file='/home1/layomi/项目代码/MGHLA/baseline_length_supertype/MGHLA/supertypes_hla_metrics.txt'
    recording_w(write_file,metrics_dict,'a+')
    
def supertype_max_min_quartile(file1,method):
    all_dict= {}
    with open(file1, 'r') as file:
        for line in file:
            key, values_str = line.split(':(')
            # Remove parentheses and split the values by comma
            values = values_str.strip()[:-1].split(', ')
            # Convert string values to float
            values = [float(val) for val in values]
            all_dict[key] = values
         
    #得到所有超型的名称，需要定位超型  
    supertype_row=[]
    supertype_file='/home1/layomi/项目代码/MMGHLA_CT/data/hla_hla/supertypes2008.txt'
    with open(supertype_file, 'r') as file:
       
        for line in file:
            values = line.strip()
            # 去除每行的换行符并添加到列表
            #data_list.append(line.strip())
            if values!='':
                values = line.strip().split()

                # 将分隔后的值转换成相应的数据类型，例如整数或浮点数
                values = [eval(value) if value.replace('.', '', 1).isdigit() else value for value in values]

                # 将这些值作为一个列表添加到外层的列表中
                supertype_row.append(values)
    
                
    supertype_set=set(d[1] for d in supertype_row)
    supertype_list=sorted(list(supertype_set))
     
    quartile_dict=dict()
    for i in range(len(supertype_list)):
        supertype=supertype_list[i]
        supertype_all_auc=[]
        out_number=0
        for key in all_dict.keys():
            if method in key and '_'+str(supertype)+'_' in key:
                if method=='smm' and 'smmpmbec' in key:
                    continue
                
                else:
                    metrics_list=list(all_dict[key])
                    #if math.isnan(metrics_list[2]) or metrics_list[0]<0.5:
                    if metrics_list[0]<=0.5:
                        out_number=out_number+1
                        supertype_all_auc.append(metrics_list[0])
                        continue
                    else:
                        supertype_all_auc.append(metrics_list[0])
            
        if supertype_all_auc!=[]:
            max_auc=max(supertype_all_auc)
            min_auc=min(supertype_all_auc)
            first_quartile = np.nanpercentile(supertype_all_auc, 25)  # 第一四分位数（25%）
            thrid_quartile = np.nanpercentile(supertype_all_auc, 75)   
            supertype_number=len(supertype_all_auc)
            #print('{}_{}:max_{}    min_{}         out_{}'.format(method,length,max_auc,min_auc,out_number))
            quartile_dict[method+'_'+str(supertype)+'_max']=round(max_auc,4)
            quartile_dict[method+'_'+str(supertype)+'_min']=round(min_auc,4)
            quartile_dict[method+'_'+str(supertype)+'_first_quartile']=round(first_quartile,4)
            quartile_dict[method+'_'+str(supertype)+'_thrid_quartile']=round(thrid_quartile,4)
            quartile_dict[method+'_'+str(supertype)+'_allnumber']=supertype_number
            quartile_dict[method+'_'+str(supertype)+'_out_number']=out_number
        else:
            print('Error')
        
    write_file='/home1/layomi/项目代码/MGHLA/baseline_length_supertype/MGHLA/supertype_quartille.txt'
    recording_w(write_file, quartile_dict,'a+')   
    
def length_max_min_quartile(file1,method):  #在选最大最小值时把预测全错的部分除去，并统计各长度预测全错部分的个数
    all_dict= {}
    with open(file1, 'r') as file:
        for line in file:
            key, values_str = line.split(':(')
            # Remove parentheses and split the values by comma
            values = values_str.strip()[:-1].split(', ')
            # Convert string values to float
            values = [float(val) for val in values]
            all_dict[key] = values
      
    quartile_dict=dict()      
    for i in range(7):
        length=i+8
        length_all_auc=[]
        out_number=0
        for key in all_dict.keys():
            if method in key and '_'+str(length)+'_' in key:
                metrics_list=list(all_dict[key])
                #if math.isnan(metrics_list[2]) or metrics_list[2]<0.5:
                if metrics_list[0]<0.5 or math.isnan(metrics_list[0]):
                    out_number=out_number+1
                    length_all_auc.append(metrics_list[0])
                    continue
                else:
                    length_all_auc.append(metrics_list[0])
            
        if length_all_auc!=[]:
            max_auc=max(length_all_auc)
            min_auc=min(length_all_auc)
            first_quartile = np.nanpercentile(length_all_auc, 25)  # 第一四分位数（25%）
            thrid_quartile = np.nanpercentile(length_all_auc, 75)   
            length_number=len(length_all_auc)
            #print('{}_{}:max_{}    min_{}         out_{}'.format(method,length,max_auc,min_auc,out_number))
            quartile_dict[method+'_'+str(length)+'_max']=round(max_auc,4)
            quartile_dict[method+'_'+str(length)+'_min']=round(min_auc,4)
            quartile_dict[method+'_'+str(length)+'_first_quartile']=round(first_quartile,4)
            quartile_dict[method+'_'+str(length)+'_thrid_quartile']=round(thrid_quartile,4)
            quartile_dict[method+'_'+str(length)+'_allnumber']=length_number
            quartile_dict[method+'_'+str(length)+'_out_number']=out_number
        else:
            print('Error')
        
    write_file='/home1/layomi/项目代码/MGHLA/baseline_length_supertype/MGHLA/length_quartille.txt'
    recording_w(write_file, quartile_dict,'a+') 
    
    
def recording_w(file1,record,w_or_a='w'):
    print('1')
    if isinstance(record,dict):
        print('True')

        with open (file1,w_or_a) as f:
            for key,value in record.items():
                print('{}:{}'.format(key,value),file=f)
        f.close()    

if __name__=='__main__':
    '''
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm/results/data_result_file/independent_alldata.txt'
    len_pos=3
    score_pos=4
    method='MGHLA'
    ic50_or_rank=False
    length_metrics_max_min(file1,len_pos,score_pos,method,ic50_or_rank)
    '''
    '''
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm/results/test_set_metrics/length/len_metrics_max_min.txt'
    method='MGHLA'
    length_max_min_quartile(file1,method)
    '''
    '''
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm/results/data_result_file/independent_alldata.txt'
    method='MGHLA'
    score_pos=-3
    ic50_or_rank=False
    supertype_metrics(method,file1,score_pos,ic50_or_rank)
    '''
    '''
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm/results/data_result_file/independent_alldata.txt'
    method='MGHLA'
    score_pos=-3
    ic50_or_rank=False
    supertype_hla_metrics(method,file1,score_pos,ic50_or_rank)
    '''
    '''
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm/results/test_set_metrics/supertype/supertypes_hla_metrics.txt'
    method='MGHLA'
    supertype_max_min_quartile(file1,method)
    '''
    
    '''
    #32dmodel进行长度和超型的性能计算
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep/results/data_result_file/independent_alldata_zong_pepenc_32dmodel.txt'
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/data_result_file/32dmodel_nodeedge34/epoch_17/independent_1.csv_alldata_zong_pepenc_32dmodel.txt'
    len_pos=3
    score_pos=4
    method='MGHLA'
    ic50_or_rank=False
    #length_metrics_max_min(file1,len_pos,score_pos,method,ic50_or_rank)
    length_metrics(file1,len_pos,score_pos,method,ic50_or_rank)
    '''
    '''
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep/results/data_result_file/independent_alldata_zong_pepenc_32dmodel.txt'
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/data_result_file/32dmodel_nodeedge34/epoch_17/independent_1.csv_alldata_zong_pepenc_32dmodel.txt'
    len_pos=3
    score_pos=4
    method='MGHLA'
    ic50_or_rank=False
    length_metrics_max_min(file1,len_pos,score_pos,method,ic50_or_rank)
    '''
    
    '''
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm/results/test_set_metrics/length_new/len_metrics_max_min.txt'
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/hpi/length_supertypes/length_new/len_metrics_max_min.txt'
    method='MGHLA'
    length_max_min_quartile(file1,method)
    '''
    '''
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/data_result_file/32dmodel_nodeedge34/epoch_17/independent_1.csv_alldata_zong_pepenc_32dmodel.txt'
    method='MGHLA'
    score_pos=4
    ic50_or_rank=False
    supertype_metrics(method,file1,score_pos,ic50_or_rank)
    '''
    '''
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/data_result_file/32dmodel_nodeedge34/epoch_17/independent_1.csv_alldata_zong_pepenc_32dmodel.txt'
    method='MGHLA'
    score_pos=4
    ic50_or_rank=False
    supertype_hla_metrics(method,file1,score_pos,ic50_or_rank)
    '''
    '''
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm/results/test_set_metrics/supertypes_new/supertypes_hla_metrics.txt'
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/hpi/length_supertypes/supertypes/supertypes_hla_metrics.txt'
    method='MGHLA'
    supertype_max_min_quartile(file1,method)
    '''
    
    #J_4metrics
    '''
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/data_result_file/structureclasstopo1_J_4/epoch_16/independent_1.csv_alldata_zong_pepenc_32dmodel.txt'
    len_pos=3
    score_pos=4
    method='MGHLA'
    ic50_or_rank=False
    #length_metrics_max_min(file1,len_pos,score_pos,method,ic50_or_rank)
    length_metrics(file1,len_pos,score_pos,method,ic50_or_rank)
    
    '''
    '''
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep/results/data_result_file/independent_alldata_zong_pepenc_32dmodel.txt'
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/data_result_file/structureclasstopo1_J_4/epoch_16/independent_1.csv_alldata_zong_pepenc_32dmodel.txt'
    len_pos=3
    score_pos=4
    method='MGHLA'
    ic50_or_rank=False
    length_metrics_max_min(file1,len_pos,score_pos,method,ic50_or_rank)
    '''
    '''
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm/results/test_set_metrics/length_new/len_metrics_max_min.txt'
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/hpi/length_supertypes/length_new_J_4/len_metrics_max_min.txt'
    method='MGHLA'
    length_max_min_quartile(file1,method)
    '''
    '''
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/data_result_file/structureclasstopo1_J_4/epoch_16/independent_1.csv_alldata_zong_pepenc_32dmodel.txt'
    method='MGHLA'
    score_pos=4
    ic50_or_rank=False
    supertype_metrics(method,file1,score_pos,ic50_or_rank)
    '''
    '''
    file1=file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/data_result_file/structureclasstopo1_J_4/epoch_16/independent_1.csv_alldata_zong_pepenc_32dmodel.txt'
    method='MGHLA'
    score_pos=4
    ic50_or_rank=False
    supertype_hla_metrics(method,file1,score_pos,ic50_or_rank)
    '''
    '''
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm/results/test_set_metrics/supertypes_new/supertypes_hla_metrics.txt'
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/hpi/length_supertypes/supertypes/supertypes_hla_metrics.txt'
    file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/hpi/length_supertypes/supertypes_new_J_4/supertypes_hla_metrics.txt'
    method='MGHLA'
    supertype_max_min_quartile(file1,method)
    '''
    '''
    #MGHLA_metrics
    
    file1='/home1/layomi/项目代码/MGHLA/results/data_result_file/fold_data_new2/epoch17/independent_1.csv'
    len_pos=3
    score_pos=4
    method='MGHLA'
    ic50_or_rank=False
    #length_metrics_max_min(file1,len_pos,score_pos,method,ic50_or_rank)
    length_metrics(file1,len_pos,score_pos,method,ic50_or_rank)
    '''
    
    '''
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep/results/data_result_file/independent_alldata_zong_pepenc_32dmodel.txt'
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/data_result_file/structureclasstopo1_J_4/epoch_16/independent_1.csv_alldata_zong_pepenc_32dmodel.txt'
    file1='/home1/layomi/项目代码/MGHLA/results/data_result_file/fold_data_new2/epoch17/independent_1.csv'
    len_pos=3
    score_pos=4
    method='MGHLA'
    ic50_or_rank=False
    length_metrics_max_min(file1,len_pos,score_pos,method,ic50_or_rank)
    '''
    '''
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm/results/test_set_metrics/length_new/len_metrics_max_min.txt'
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/hpi/length_supertypes/length_new_J_4/len_metrics_max_min.txt'
    file1='/home1/layomi/项目代码/MGHLA/baseline_length_supertype/MGHLA/len_metrics_max_min.txt'
    method='MGHLA'
    length_max_min_quartile(file1,method)
    '''
    '''
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/data_result_file/structureclasstopo1_J_4/epoch_16/independent_1.csv_alldata_zong_pepenc_32dmodel.txt'
    file1='/home1/layomi/项目代码/MGHLA/results/data_result_file/fold_data_new2/epoch17/independent_1.csv'
    method='MGHLA'
    score_pos=4
    ic50_or_rank=False
    supertype_metrics(method,file1,score_pos,ic50_or_rank)
    '''
    '''
    #file1=file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/data_result_file/structureclasstopo1_J_4/epoch_16/independent_1.csv_alldata_zong_pepenc_32dmodel.txt'
    file1='/home1/layomi/项目代码/MGHLA/results/data_result_file/fold_data_new2/epoch17/independent_1.csv'
    method='MGHLA'
    score_pos=4
    ic50_or_rank=False
    supertype_hla_metrics(method,file1,score_pos,ic50_or_rank)
    '''
    
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm/results/test_set_metrics/supertypes_new/supertypes_hla_metrics.txt'
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/hpi/length_supertypes/supertypes/supertypes_hla_metrics.txt'
    #file1='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight_pep_structure/results/hpi/length_supertypes/supertypes_new_J_4/supertypes_hla_metrics.txt'
    file1='/home1/layomi/项目代码/MGHLA/baseline_length_supertype/MGHLA/supertypes_hla_metrics.txt'
    method='MGHLA'
    supertype_max_min_quartile(file1,method)
    
    