from enum import Enum
from operator import length_hint
import re
from turtle import dot
from get_source_des import GET_IN_OUT
import numpy as np
import os

path1 = os.path.dirname(os.path.abspath(__file__))
#Type of edge
class Edge(Enum):
#control flow：
    control = 1
#data flow：
    data = 2
#call：
    fun_call = 3
#access：
    load_store = 4
#jump:
    jump = 5

#The instruction list is obtained by each program
ins_list = ['alloca','store','load','call','br','ret','icmp','add','sub','bitcast', 'getelementptr', 'srem', 'sext', 'mul', 'select', 'phi'] 
#Predefined number of bits of instruction
ins_width = {'alloca': 64, 'store':32, 'load':32, 'call':64, 'br': 1, 'ret': 32, 'icmp': 1, 'add': 32, 'sub': 32, 'bitcast': 64, 'getelementptr': 64, 'srem': 32, 'sext': 64, 'mul': 64, 'select':32, 'phi': 32}

#LLVM instruction type list
Ins_type = {'ret': 'terminator', 'br': 'terminator', 'switch': 'terminator', 'indirectbr': 'terminator', 'invoke': 'terminator', 'resume': 'terminator', 'unreachable': 'terminator', 'cleanupret': 'terminator', 'catchret': 'terminator', 'catchswitch': 'terminator', 'call': 'terminator', 'add': 'int_binary', 'sub': 'int_binary', 'mul': 'int_binary', 'udiv': 'int_binary', 'sdiv': 'int_binary', 'urem': 'int_binary', 'srem': 'int_binary', 'fadd': 'float_binary', 'fsub': 'float_binary', 'fmul': 'float_binary', 'fdiv': 'float_binary', 'frem': 'float_binary', 'shl': 'logic', 'lshr': 'logic', 'ashr': 'logic', 'and': 'logic', 'or': 'logic', 'xor': 'logic', 'alloca': 'Memory', 'load': 'Memory', 'store': 'Memory', 'getelementptr': 'Memory', 'fence': 'Memory', 'trunc': 'cast_op', 'zext': 'cast_op', 'sext': 'cast_op', 'fptout': 'cast_op', 'ptrtoint': 'cast_op', 'inttoptr': 'cast_op', 'bitcast': 'cast_op', 'addrspacecast': 'cast_op', 'icmp': 'compare', 'fcmp': 'compare', 'phi': 'other', 'select': 'other', 'vaarg': 'other'}


fun_name = "" #Function name
str2 = "BB" #Basic block abbreviation name
Instruction_type = {} #Instruction type

#Remove null
def not_empty(s):
    return s and s.strip()

#Get the number of program instructions in LLVM
dict_SDC = {} 
with open(path1+"\\F_B_I.dot","r") as f:     
    for line in f:
        elem = list(filter(not_empty,re.split("\n|\$",line)))
        #print(elem)
        dict_SDC[elem[0]] = elem[1]

#Get the number of bits and error rate of the instruction
def get_Ins_SDC(filename1 , filename2):    
    Ins_SDC = np.loadtxt(filename1, skiprows=1, dtype=str)
    Ins_other = np.loadtxt(filename2, skiprows=1, dtype=str)
    part_SDC = {}
    for i in range(Ins_SDC.shape[0]): #Get the width and error rate of each instruction in the destination register
        if Ins_SDC[i][1] not in part_SDC.keys(): #If there is no such instruction, update it
            temp = [0]*3
            temp[0] = int(Ins_SDC[i][3])
            temp[1] = int(Ins_SDC[i][3])
            temp[2] = int(Ins_SDC[i][4])
            part_SDC[Ins_SDC[i][1]] = temp
        else: #Sum
            part_SDC[Ins_SDC[i][1]][1] += int(Ins_SDC[i][3])
            part_SDC[Ins_SDC[i][1]][2] += int(Ins_SDC[i][4])
    for i in range(Ins_other.shape[0]): #Get the width and error rate of each instruction in the source register
        if Ins_other[i][1] not in part_SDC.keys():
            temp = [0]*3
            temp[0] = int(Ins_other[i][3])
            temp[1] = int(Ins_other[i][3])
            temp[2] = int(Ins_other[i][4])
            part_SDC[Ins_other[i][1]] = temp
        else:
            part_SDC[Ins_other[i][1]][1] += int(Ins_other[i][3])
            part_SDC[Ins_other[i][1]][2] += int(Ins_other[i][4])
    part_Ins_SDC = {} #Store instruction width and error rate according to instruction number
    keys = list(part_SDC.keys())
    for i in range(len(keys)): #Average error rate of each instruction
        temp = [0]*2
        temp[0] = part_SDC[keys[i]][0]
        temp[1] = round((part_SDC[keys[i]][2] / part_SDC[keys[i]][1]), 4)
        part_Ins_SDC[keys[i]] = temp
    SDC = {}
    SDC_flag = {}
    keys = list(part_Ins_SDC.keys())
    for i in range(len(keys)): #Average error rate of each instruction
        if dict_SDC[keys[i]] not in SDC.keys():
            SDC[dict_SDC[keys[i]]] = part_Ins_SDC[keys[i]]
            SDC_flag[dict_SDC[keys[i]]] = 1
        else:#Error rate of duplicate instruction statistics 
            SDC[dict_SDC[keys[i]]][1] += part_Ins_SDC[keys[i]][1]
            SDC_flag[dict_SDC[keys[i]]] += 1
    keys = list(SDC_flag.keys())
    for i in range(len(keys)): #Limit the width of floating points 
        if SDC_flag[keys[i]] > 1:
            SDC[keys[i]][1] = round(SDC[keys[i]][1]/SDC_flag[keys[i]], 4)
    return SDC

#Fault injection result file
filename1 = path1+'\\cycle_result.txt'
filename2 = path1+"\\result_other.txt"
#Get instruction bits and error rate
Ins_SDC = get_Ins_SDC(filename1, filename2)

#Basic block information file
BB_filename = path1+ "\\BB.dot"
dict_BB = {} #Basic block set, as {'BB0': [13, 0, 1]},
BB_edge = [] #Edge between basic blocks, as ['BB0', 'BB1'], ['BB4', 'BB1'],
#Get the attributes of the base block
def get_BB(filename):
    with open(filename,"r") as f:
        for line in f:
            elem = list(filter(not_empty,re.split("\n| ",line)))
            if "BB" in elem[0]:
                str2 = list(filter(not_empty,re.split("BB", elem[0])))
                dict_BB[str2[0]] = []
                curbb = str2[0]
                pred = 0
                succ = 0
                dict_BB[curbb].append(int(elem[1])) #Number of instructions contained
            elif elem[0] == "pred:": #Precursors
                for i in range(1,len(elem)): 
                    edge = []
                    edge.append(elem[i])
                    edge.append(curbb)
                    BB_edge.append(edge)
                    pred += 1
                dict_BB[curbb].append(pred)
            elif elem[0] == "succ:": #Successors
                for i in range(1, len(elem)):
                    edge = []
                    edge.append(curbb)
                    edge.append(elem[i])
                    BB_edge.append(edge)
                    succ += 1
                dict_BB[curbb].append(succ)
            elif elem[0] == "funcall": #Jump of basic block during function call
                for i in range(1, len(elem)):
                    lines = list(filter(not_empty, re.split("->", elem[i])))
                    edge = []
                    edge.append(lines[0])
                    edge.append(lines[1])
                    BB_edge.append(edge)
                              
get_BB(BB_filename)
#Get the number of each instruction entity
def get_index(s):
    for i in range(len(ins_list)):
        if s == ins_list[i]:
            return i

dict_Ins = {}   #Instruction node
dict_edge = []  #Edge between nodes, as [["V1","1","V2"]]
node_feature = {} #Instruction Attribute Table, as {"0":[bit,Precursors,Successors,Number of operands,Type,Basic block,Function,error rate]} 

ins_num = 0
relationship = 0
flow_flag = 0

#Judge whether the instruction node exists
def get_key(value):
    keys = list(dict_Ins.keys())
    values = list(dict_Ins.values())
    for i in range(len(values)):
        if value == values[i]:
            return i
    return -1

#access relation instruction
alloca_load  = ['alloca','load']
#Get the contents of the instruction file obtained by LLVM
with open (path1+"\\Ins_g.dot","r") as f:
    for line in f:
        elem = list(filter(not_empty,re.split("->|\n|{|\"|;",line)))
        first = list(filter(None,re.split("_| ",elem[0])))
        #Get the current function name
        if len(first) == 3 and first[1] == 'cluster' and str2 not in first[2]:
            fun_name = first[2]
        #Get the current basic block
        elif first[0] == 'label' and str2 in first[2]:
            bb_name =  first[2]
        elif elem[0] == "dataflow": #Judge the relation between various instructions
            flow_flag = 2
            continue
        elif elem[0] == "controlflow":
            flow_flag = 1
            continue
        elif elem[0] == "bb_call":
            flow_flag = 5
        elif elem[0] == "fun_call":
            flow_flag = 3
        elif len(elem) >= 2 or ("ret" in elem[0]):
            #print(elem)
            temp = []
            rel = [] #Multiple relations between nodes
            rel.append(flow_flag)
            relationship = flow_flag
            for i in range(0,len(elem)):
                features = [0]*8 #Instruction Attribute Table
                features[6] = fun_name 
                features[5] = bb_name 
                features[7] = -1 #error rate
                in_out = []
                elem1 = list(filter(not_empty,re.split(",|[ ]+",elem[i])))
                #Get the operation code of each instruction
                if (len(elem1) > 3 and elem1[3] in ins_list):
                    features[4] = Ins_type[elem1[3]] #Type
                    features[0] = ins_width[elem1[3]] #Bit
                    input, output = GET_IN_OUT(elem1[3],elem[i]).get_input_output() #The nuumber of operands
                    features[3] = len(input) + len(output) #Number of predecessors and successors
                    index = get_index(elem1[3])
                    Ins_index = get_key(elem[i])
                    if Ins_index == -1:
                        str1 = str(ins_num)
                        ins_num = ins_num + 1
                        Instruction_type[str1] = elem1[3]
                        dict_Ins[str1] = elem[i]
                        node_feature[str1] = features
                    else:
                        str1 = list(dict_Ins.keys())[Ins_index]
                    temp.append(str1)
                    #Is it an access instruction
                    if ins_list[index] in alloca_load and flow_flag == 2 and len(temp) == 1:
                        rel.append(4)
                        relationship = 4
                elif (len(elem1) >= 3 and elem1[1] in ins_list):
                    features[4] = Ins_type[elem1[1]]
                    features[0] = ins_width[elem1[1]]
                    input, output = GET_IN_OUT(elem1[1],elem[i]).get_input_output()
                    features[3] = len(input) + len(output)
                    index = get_index(elem1[1])
                    Ins_index = get_key(elem[i])
                    if Ins_index == -1:
                        str1 = str(ins_num)
                        ins_num = ins_num + 1
                        Instruction_type[str1] = elem1[1]
                        dict_Ins[str1] = elem[i]
                        node_feature[str1] = features
                    else:
                        str1 = list(dict_Ins.keys())[Ins_index]
                    temp.append(str1)
                    if ins_list[index] in alloca_load and flow_flag == 2 and len(temp) == 1:
                        rel.append(4)
                        relationship = 4
            #Get the relations between instruction nodes
            if len(temp) == 2:
                for i in range(len(rel)):
                    flag = []
                    flag.append(temp[0])
                    flag.append(temp[1])
                    flag.insert(1,rel[i])
                    dict_edge.append(flag)

def pre_suc():
    for i in range(len(dict_edge)):
        node_feature[dict_edge[i][0]][2] += 1 #successors + 1
        node_feature[dict_edge[i][2]][1] += 1 #predecessors + 1
#Get instruction entity
def get_node():
    return dict_Ins
#Get edges between instruction entities
def get_edge():
    return dict_edge

True_Vulnerability = {} #The true error rate of instruction
#Get the attributes of instruction
def get_features():
    pre_suc()
    keys = list(Ins_SDC.keys())
    ins_keys = list(dict_Ins.keys())
    ins_value = list(dict_Ins.values())
    for i in range(len(keys)):
        index = ins_value.index(keys[i])
        node_feature[ins_keys[index]][0] = Ins_SDC[keys[i]][0] #Update instruction bits according to fault injection results
        node_feature[ins_keys[index]][7] = Ins_SDC[keys[i]][1] #Update error rate according to fault injection results
        True_Vulnerability[ins_keys[index]] = Ins_SDC[keys[i]][1]
    keys = list(node_feature.keys())
    return node_feature

#Get basic block information
def get_BB_info():
    return dict_BB, BB_edge
#features = get_features()
#Store instruction information
path2 = path1 + '\\Instruction_type.npy'
np.save(path2, Instruction_type)

path2 = path1 + '\\Vulnerability_true.npy'
np.save(path2, True_Vulnerability)

if __name__ == '__main__':
    #features = get_features()
    node = get_node()
    print(node)