#!/usr/bin/env python
# coding: utf-8

from collections import OrderedDict
#import pandas as pd
def readData():
    #z = pd.read_csv("data.csv",delimiter = ";")
    cnt = 0
    fp = open("breast-cancer-wisconsin.data.data","r")   
    dict_data = {} 
    dict_count = OrderedDict()
    labels = []
    len1 = 0
    for line in fp:
        if cnt == 0:
            val = line.strip('\n').split(",")
            val1 = val
            #label = z[val[len(val) - 1]].unique() 
            #print(label)
        if cnt >= 1:
            val = line.strip('\n').split(",")
            dict_data[cnt - 1] = val   #{val[i]: 1}
            if(val[len(val) - 1]) not in labels:
                labels.append(val[len(val) - 1])
        cnt += 1
    fp.close()
    return dict_data, labels
    


# In[713]:


import math
def Entropy(dict_data):
    labels = {}
    for val in dict_data.values():
        if(val[len(val) - 1]) in labels.keys():
            labels[val[len(val) - 1]] = labels[val[len(val) - 1]] + 1
        else:
            labels[val[len(val) - 1]] = 1 
    sum_label = sum(labels.values())
    entropy = 0
    for key,val in labels.items():
        labels[key] = (float(labels[key])/sum_label)
        entropy = entropy + -1*(labels[key] * (math.log(labels[key])/math.log(2)))
    return entropy


# In[714]:


#This function returns the unique attribute values for the attribute we split upon
def splitting_index_values(dict_data,splitting_index):
    unique_val = []
    for val in dict_data.keys():
        if dict_data[val][splitting_index] not in unique_val:
            unique_val.append(dict_data[val][splitting_index])
    return unique_val


# In[715]:


#THis function returns the partitoned data for the attribute values we split upon
from collections import OrderedDict
def get_partition(dict_data,attr_val,splitting_index):
    data = OrderedDict()
    splitting_index = int(splitting_index)
    val = 0
    for key in dict_data.keys():
        if dict_data[key][splitting_index] in attr_val:
            data[val] = {}
            data[val] = dict_data[key]
            data[val].remove(data[val][splitting_index])
            val = val + 1
    return data


# In[716]:


#THis function returns the partitoned data for the attribute values we split upon. Use it for only continuous data
from collections import OrderedDict
def get_partition_continuous(dict_data,splitting_index,split_mid_pt):
    splitting_index = int(splitting_index)
    val = 0
    val_more = 0
    d_less = OrderedDict()
    d_more = OrderedDict()
    for key in dict_data.keys():
        if float(dict_data[key][splitting_index]) <= split_mid_pt:
            d_less[val] = {}
            d_less[val] = dict_data[key]
            #print(dict_data[key])
            d_less[val].remove(d_less[val][splitting_index])
            val = val + 1
        else:
            d_more[val_more] = {}
            d_more[val_more] = dict_data[key]
            d_more[val_more].remove(d_more[val_more][splitting_index])
            val_more = val_more + 1
    #print("d=",d_less,d_more)
    return d_less,d_more


# In[717]:


#This function returns the gini index for all the unique combinations for each attribute
def gini_index(dict_data,attr_set,labels,index):
    gini_d = 0.0
    gini_attr = 0.0
    sum_attr = 0.0
    sum_not_attr = 0.0
    sum_label = 0.0
    sum_not_label = 0.0
    len_data = float(len(dict_data))
    #print(labels)
    for label_val in labels:
        p = -1 * [dict_data[row][-1] for row in dict_data.keys()].count(label_val)/len_data
        p_attr = -1 * [row[-1] for row in (dict_data[val] for val in dict_data.keys() if dict_data[val][index] in attr_set)].count(label_val)
        p_not_attr = -1 * [row[-1] for row in (dict_data[val] for val in dict_data.keys() if dict_data[val][index] not in attr_set)].count(label_val)
        sum_label += (-1 * p_attr)
        sum_not_label += (-1 * p_not_attr)
        sum_attr += float((p_attr * p_attr))
        sum_not_attr += float((p_not_attr * p_not_attr)) 
        gini_d += p*p
    if(sum_label != 0):
        sum_attr = ((sum_label/len_data) * (1 - ((sum_attr)/(sum_label*sum_label))))
    if((sum_not_label) != 0):
        sum_not_attr = (((sum_not_label)/len_data) * (1 - ((sum_not_attr)/((sum_not_label)*(sum_not_label)))))
        #print(sum_not_attr)
    gini_attr = sum_attr + sum_not_attr
    gini_d = 1 - gini_d
    return gini_d - gini_attr
      


# In[718]:


#THis function returns the gini index for continuous attributes
def Gini_Index_Continuous(dict_data,labels,index):
    split_md_pt = 0.0
    min_con_gini_ind = -10.0
    len_data = float(len(dict_data))
    sorted_data_list = sorted(dict_data.values(), key=lambda e: e[index], reverse = True)
    sorted_data = OrderedDict()
    for i in range(0,len(sorted_data_list)):
        sorted_data[i] = {}
        sorted_data[i] = sorted_data_list[i]
    unique_val_list = list(map(float,list(set(row[index] for row in (sorted_data[val] for val in sorted_data.keys())))))
    for i in range(0, len(unique_val_list)-1):
        mid_pt = (unique_val_list[i] + unique_val_list[i+1])/2
        d1_less = OrderedDict({k:v for (k,v) in sorted_data.items() if float(sorted_data[k][index]) <= mid_pt})
        len_less = float(len(d1_less))
        d1_more = {k:v for (k,v) in sorted_data.items() if float(sorted_data[k][index]) > mid_pt}
        len_more = float(len(d1_more))
        sum_label = 0.0
        sum_not_label = 0.0
        sum_attr = 0.0
        sum_not_attr = 0.0
        gini_d = 0.0
        for label_val in labels:
            p = -1 * [dict_data[row][-1] for row in dict_data.keys()].count(label_val)/len_data
            p_attr = -1 * [row[-1] for row in (dict_data[val] for val in dict_data.keys() if float(dict_data[val][index]) <= mid_pt)].count(label_val)
            p_not_attr = -1 * [row[-1] for row in (dict_data[val] for val in dict_data.keys() if float(dict_data[val][index]) > mid_pt)].count(label_val)
            sum_label += (-1 * p_attr)
            sum_not_label += (-1 * p_not_attr)
            sum_attr += float((p_attr * p_attr))
            sum_not_attr += float((p_not_attr * p_not_attr)) 
            gini_d += p*p
        if(sum_label != 0):
            sum_attr = ((sum_label/len_data) * (1 - ((sum_attr)/(sum_label*sum_label))))
            #print(sum_attr)
        if((sum_not_label) != 0):
            sum_not_attr = (((sum_not_label)/len_data) * (1 - ((sum_not_attr)/((sum_not_label)*(sum_not_label)))))
            #print(sum_not_attr)
        gini_attr = sum_attr + sum_not_attr
        gini_d = 1 - gini_d 
        gini_index = gini_d - gini_attr
        #print(gini_index)
        if(min_con_gini_ind < gini_index):
            min_con_gini_ind = gini_index
            split_md_pt = mid_pt
    #print(min_con_gini_ind)
    return min_con_gini_ind,split_md_pt
    


# In[719]:


#THis function returns the attribute value & index of the attribute with minimum gini val
import itertools
def get_subset(dict_data,labels):
    #print("D=",dict_data)
    min_gini_index = -10.0
    split_attr_val = []
    #print("gini ", len(dict_data[0]))
    for i in range (len(dict_data[0])-1):
        if(dict_data[0][i].replace('.','').isdigit()):
                cont_gini_index,split_mid_pt_gini = Gini_Index_Continuous(dict_data,labels,i)
                if(min_gini_index < cont_gini_index):
                    min_gini_index = cont_gini_index
                    iscontinuous_gini = True
                    split_index_gini = i
        else:
            attr_val = splitting_index_values(dict_data,i)
            for j in range(1,len(attr_val)):
                z = list(itertools.combinations(attr_val,j))
                for k in range(0,len(z)):
                    gini_ind = gini_index(dict_data,z[k],labels,i)
                    if(min_gini_index < gini_ind):
                        min_gini_index = gini_ind
                        split_attr_val = z[k]
                        iscontinuous_gini = False
                        split_index_gini = i
    return min_gini_index,split_attr_val,split_index_gini,iscontinuous_gini,split_mid_pt_gini
        


# In[720]:


#print(sorted(dict_data.values(), key=lambda e: e[0], reverse = True))
#s = list(set(row[0] for row in (dict_data[val] for val in dict_data.keys())))
#d1 = {k:v for (k,v) in dict_data.items() if dict_data[k][0] == '2'}

#This function calculates info gain for continuous data and returns the split point with best info gain
import math
from collections import OrderedDict
def continuous_info_gain(dict_data,index,entropy,labels):
    sum_less = 0.0
    sum_more = 0.0
    split_mid_pt = 0.0
    split_mid_pt_gain_ratio = 0.0
    min_cont_info = 100.0
    max_gain_ratio = 0.0
    len_data = float(len(dict_data))
    sorted_data_list = sorted(dict_data.values(), key=lambda e: e[index], reverse = True)
    sorted_data = OrderedDict()
    i = 0
    for i in range(0,len(sorted_data_list)):
        sorted_data[i] = {}
        sorted_data[i] = sorted_data_list[i]
    unique_val_list = list(map(float,list(set(row[index] for row in (sorted_data[val] for val in sorted_data.keys())))))
    #print("$$$$$$$$$$$$ ", unique_val_list)
    if(len(unique_val_list)==1):
        split_mid_pt = unique_val_list[0]
        split_mid_pt_gain_ratio = unique_val_list[0]
    for i in range(0, len(unique_val_list)-1):
        mid_pt = (unique_val_list[i] + unique_val_list[i+1])/2
        d1_less = OrderedDict({k:v for (k,v) in sorted_data.items() if float(sorted_data[k][index]) <= mid_pt})
        len_less = float(len(d1_less))
        d1_more = {k:v for (k,v) in sorted_data.items() if float(sorted_data[k][index]) > mid_pt}
        #print("d=",d1_less)
        #print("d2=",d1_more)
        len_more = float(len(d1_more))
        sum_less = 0.0
        sum_more = 0.0
        tot_sum_less = 0.0
        tot_sum_more = 0.0
        split_info = 0.0
        '''for label_val in labels:
            info_less = float([d1_less[row][-1] for row in d1_less.keys()].count(label_val))/len_less
            info_more = float([d1_more[row][-1] for row in d1_more.keys()].count(label_val))/len_more
            #tot_sum_less += float([d1_less[row][-1] for row in d1_less.keys()].count(label_val))
            #tot_sum_more += float([d1_more[row][-1] for row in d1_more.keys()].count(label_val))
            if(info_less != 0):
                sum_less += (-1 * info_less * (math.log(info_less)/math.log(2)))
            if(info_more != 0):
                sum_more += (-1 * info_more * (math.log(info_more)/math.log(2)))'''
                
        if(len(d1_less) != 0):
            #print(len(d1_less),len(dict_data))
            tot_sum_less = (-1*(float(len(d1_less))/len(dict_data))) * math.log(float(len(d1_less))/len(dict_data))/math.log(2)
        if(len(d1_more) != 0):
            tot_sum_more = (-1*(float(len(d1_more))/len(dict_data))) * math.log(float(len(d1_more))/len(dict_data))/math.log(2)
        
        #print("tot=",tot_sum_less,tot_sum_more)
        cont_info = ((len_less/len_data)*sum_less + (len_more/len_data)*sum_more)
        split_info = tot_sum_more + tot_sum_less
        gain_ratio = ((entropy - cont_info)/split_info)
        if(cont_info < min_cont_info):
            min_cont_info = cont_info
            split_mid_pt = mid_pt
        else:#
            #print("$$$$$$$$$$$$$$$$$$$ ", mid_pt)
            split_mid_pt = mid_pt
        if(gain_ratio > max_gain_ratio):
            max_gain_ratio = gain_ratio
            split_mid_pt_gain_ratio = mid_pt
        #print("MG=",gain_ratio,mid_pt)
    return entropy - min_cont_info,split_mid_pt,max_gain_ratio,split_mid_pt_gain_ratio
    


# In[721]:


#This function caculates the info gain & gain ratio for categorical data and checks if the data is continuous then continuous info
#gain function is called. Function returns the following values:
#max_gain - Max info gain
#splitting_index - Index(attribute index) with max info gain
#max_gain_ratio - Maximum gain ratio(for categorical data)
#gain_ratio_splitting_index - Index(attribute index) with max gain ratio
#iscontinuous - Tells whether the split pt is categorical or continuous
#split_mid_pt - Mid pt of the split. Used in case of continuous data
from collections import OrderedDict
import math
def Info_gain(dict_data,entropy,labels):
    info_attr = 0.0
    splitting_index = 0
    sum_attr = 0.0
    max_gain = 0.0
    max_gain_ratio = 0.0
    gain_ratio = 0.0
    gain_ratio_splitting_index = 0
    gain = 0
    split_mid_pt = 0.0
    iscontinuous = False
    iscontinuous_gain_ratio = False
    key_val = next(iter(dict_data))
    #print("key=",len(dict_data[0]))
    dict_val_count = OrderedDict()
    for i in range (len(dict_data[0])-1):
        for key in dict_data.keys():
            isdiscrete = True
            if(dict_data[key][i].replace('.','').isdigit()):
                cont_info_gain,split_mid_pt,gain_ratio_cont,split_mid_pt_gain_ratio = continuous_info_gain(dict_data,i,entropy,labels)
                #print("mid point is ", split_mid_pt)
                iscontinuous = True
                iscontinuous_gain_ratio = True
                isdiscrete = False
                if(cont_info_gain > max_gain):
                    max_gain = cont_info_gain
                    splitting_index = i
                if(gain_ratio_cont > max_gain_ratio):
                    max_gain_ratio = gain_ratio_cont
                    gain_ratio_splitting_index = i
            else:
                if (dict_data[key][i]) not in dict_val_count.keys():
                    dict_val_count[dict_data[key][i]] = {}
                if(dict_data[key][len(dict_data[0])-1]) not in dict_val_count[dict_data[key][i]].keys():
                    dict_val_count[dict_data[key][i]][dict_data[key][len(dict_data[0]) - 1]] = 1
                else:
                    dict_val_count[dict_data[key][i]][dict_data[key][len(dict_data[0])-1]] += 1
        if(isdiscrete == True):
            total_info_attr = 0
            split_info_attr = 0
                #print(dict_val_count)
            for key in dict_val_count.keys():
                    #print(dict_val_count[key])
                info_attr = 0
                sum_attr = sum((dict_val_count[key].values()))
                val_split = (float(sum_attr)/len(dict_data))
                split_info_attr += (-1 * val_split * (math.log(val_split)/math.log(2)))
                for key_attr in dict_val_count[key].keys():
                    if(sum_attr != 0):
                        val = float(dict_val_count[key][key_attr])/sum_attr
                        info_attr += (-1*val * (math.log(val)/math.log(2)))
                total_info_attr += ((float(sum_attr)/len(dict_data)) * info_attr)
                    #print(total_info_attr)
            gain = entropy - total_info_attr
            if(split_info_attr != 0):
                gain_ratio = gain/split_info_attr
                #print(gain)
                #print(gain_ratio)
                dict_val_count.clear()
            if(gain > max_gain):
                max_gain = gain
                splitting_index = i
                iscontinuous = False
            if(gain_ratio > max_gain_ratio):
                max_gain_ratio = gain_ratio
                gain_ratio_splitting_index = i
                iscontinuous_gain_ratio = False
            
    return max_gain,splitting_index ,max_gain_ratio,gain_ratio_splitting_index,iscontinuous,iscontinuous_gain_ratio,split_mid_pt,split_mid_pt_gain_ratio      


# In[722]:


class Tree_Node:
    def __init__(self,question,parent_decision):
        self.name = question
        self.parent_decision = parent_decision
        self.children = {}
        self.isconctinuous = False
        self.split = None


# In[723]:


import copy
def gen_Tree(dict_org, labels, attrList, parent_decision, function):#does labels get updated for partitions?
    dict_copy = dict_org.copy()
    attrList_copy = attrList
    #print("recur")
    #print(attrList_copy)
    N = Tree_Node(None, parent_decision)
    x=0
    tempClass = dict_org[x][-1]
    while(dict_org[x][-1] == tempClass):
        if(x == len(dict_org)-1):
            N.name = tempClass
            return N
        x+=1
    if(len(attrList_copy)==0):
        N.name = getMajorityClass(dict_org)
        return N
    entropy=Entropy(dict_org)
    if function == 0:#info gain
        max_gain,splitting_index ,max_gain_ratio,gain_ratio_splitting_index,iscontinuous,iscontinuous_gain_ratio,split_mid_pt,split_mid_pt_gain_ratio = Info_gain(dict_org,entropy,labels)
    elif function == 1:#gain ratio
        max_gain,splitting_index ,max_gain_ratio,gain_ratio_splitting_index,iscontinuous,iscontinuous_gain_ratio,split_mid_pt,split_mid_pt_gain_ratio = Info_gain(dict_org,entropy,labels)
        splitting_index = gain_ratio_splitting_index
        iscontinuous = iscontinuous_gain_ratio
        split_mid_pt = split_mid_pt_gain_ratio
        #print(iscontinuous_gain_ratio)
    elif function == 2:#gini
        min_gini_index,split_attr_val,splitting_index,iscontinuous,split_mid_pt = get_subset(dict_org,labels)
    N.name = attrList_copy[splitting_index]
    N.iscontinuous = iscontinuous
    N.split = split_mid_pt
    attrList_copy.pop(splitting_index)#attributes are not being looked at explicitly in ths function
    if(iscontinuous):
        thePartitions = [0,0]
        thePartitions[0], thePartitions[1] = get_partition_continuous(dict_org, splitting_index, split_mid_pt)
        unique_split_val = [("<= " + str(split_mid_pt)), ("> " + str(split_mid_pt))]
    else:
        thePartitions = []
        unique_split_val = splitting_index_values(dict_org,splitting_index)
        for x in unique_split_val:
            thePartitions.append(get_partition(dict_org, x, splitting_index))
    count1=0
    for x in thePartitions:
        if(len(x) == 0):
            N.children[unique_split_val[count1]] = Tree_Node(getMajorityClass(dict_copy), unique_split_val[count1])
        else:
            N.children[unique_split_val[count1]] = gen_Tree(x, labels, attrList_copy, unique_split_val[count1], function)
        count1+=1
    return N
    
def getMajorityClass(dict_org):
    theClasses = {}
    majority = ""
    for x in dict_org.values():
        if(x[-1] not in theClasses):
            theClasses[x[-1]] = 1
        else:
            theClasses[x[-1]] += 1
    y=0
    for x in theClasses:
        if(theClasses[x] > y):
            y = theClasses[x]
            majority = x
    return majority


def printTree(Node):
    #print("---")
    #print("Current question is ", Node.name)
    #print("Previous decision is ", Node.parent_decision)
    if len(Node.children)==0:
        #print("this is the leaf result")
        return
    for child in Node.children:
        printTree(Node.children[child])
        


import random
def getLearningSetAndTestingSet(org_dict, ratio):
    random.shuffle(org_dict)
    LearningSetLength = int(len(org_dict)*ratio)
    LearningSet = {}
    TestingSet = {}
    for x in range(len(org_dict)):
        if x <= LearningSetLength:
            LearningSet[x] = org_dict[x]
        else:
            TestingSet[x-LearningSetLength-1] = org_dict[x]
    return LearningSet, TestingSet
        
def TestWithTree(TestEntry, decisionTree, majority):
    if len(decisionTree.children) == 0:
        return decisionTree.name
    else:
        #print("testing entry is ", TestEntry)
        if decisionTree.iscontinuous == False:         
            if TestEntry[decisionTree.name] in decisionTree.children.keys():                
                return TestWithTree(TestEntry, decisionTree.children[TestEntry[decisionTree.name]], majority)
            else:               
                return majority
            
        else:            
            if Decimal(TestEntry[decisionTree.name]) <= decisionTree.split:
                #print(decisionTree.children[("<= " + str(decisionTree.split))])
                return TestWithTree(TestEntry, decisionTree.children[("<= " + str(decisionTree.split))], majority)
            else:
                #print(decisionTree.children[("> " + str(decisionTree.split))])
                return TestWithTree(TestEntry, decisionTree.children[("> " + str(decisionTree.split))], majority)
    
    


# In[728]:


from decimal import Decimal
import os
import time
import psutil
start_time = time.time()
dict_org, labels = readData()
entropy = Entropy(dict_org)
LearningSet, TestingSet = getLearningSetAndTestingSet(dict_org, 0.7)
attrNum=len(LearningSet[0])-1
#print(attrNum)
attrList = []
for x in range(attrNum):
    attrList.append(x)
theTree=gen_Tree(LearningSet, labels, attrList, None, 1)
printTree(theTree)

success=0.0
for x in TestingSet:
    if TestWithTree(TestingSet[x], theTree, majority) == TestingSet[x][-1]:
        success+=1
        
accuracy = success/len(TestingSet)
print("accuracy is ", accuracy)
process = psutil.Process(os.getpid())
print("Exec time= ",time.time() - start_time)
print("Mem= ",process.memory_info().rss/1000000)


    

