import numpy as np
import scanpy as sc
import math
from math import ceil

def conflict_avoid(row_len):
    latency=4   #9         #需要的延时
    color=np.zeros(latency,int)     #相当于涂色问题了
    for i in range(len(row_len)):
        if row_len[i]!=0:
            min_ind=np.argmin(color)
            color[min_ind]+=row_len[i]
    return max(color)*latency

def comp(row_len,PE_num,balance_flag,conflict_flag):

    task_queue=[]
    for i in range(PE_num):
        task_queue.append([])
    if balance_flag:
        task_lenth=np.zeros(PE_num,int)
        for i in range(len(row_len)):
            if row_len[i]!=0:
                min_ind=np.argmin(task_lenth)
                task_lenth[min_ind]+=row_len[i]
                task_queue[min_ind].append(row_len[i])
    else:
        for i in range(len(row_len)):
            task_queue[i%PE_num].append(row_len[i])
    
    task_final_time=[]
    if conflict_flag:
        for i in range(PE_num):
            task_final_time.append(conflict_avoid(task_queue[i]))
    else:
        for i in range(PE_num):
            task_final_time.append(sum(task_queue[i]))
    
    return max(task_final_time)     #返回执行时间最长的

def load_balance_sim(data,row,col):

    sparsity=data.nnz/row/col

    M0=4096*24     #每个计算单元组的行长度
    K0=1024
    N0=4

    PE_tile_num=16           #先不改  看看block和负载均衡的影响

    M_blocknum=8
    N_blocknum=2


    row_round=int((row-1)/ M0/M_blocknum)+1      #注意
    row_block=row_round*M_blocknum         #划分的小块数量  和round区分在于一次任务中处理多个小块
    col_round=int((col-1)/K0)+1

    submat_row_len=[]           #表示每行有多少个非零元
    submat_nnz_num=[]
    for i in range(row_block*col_round):
        submat_row_len.append(np.zeros(M0,int))
        submat_nnz_num.append(0)

    for i in range(row):
        row_block_ite=int(i/(M0*M_blocknum))
        row_ite=row_block_ite*M_blocknum+i%(M0*M_blocknum)%M_blocknum
        rela_row=int((i%(M0*M_blocknum))/M_blocknum)
        for j in range(data.indptr[i],data.indptr[i+1]):
            col_ite=int(data.indices[j]/K0)
            rela_col=data.indices[j]%K0
            submat_row_len[row_ite*col_round+col_ite][rela_row]+=1
            submat_nnz_num[row_ite*col_round+col_ite]+=1

    ideal=np.zeros(row_block*col_round,int)         #完全的负载均衡
    has_balance=np.zeros(row_block*col_round,int)   #不考虑冲突的我们的性能
    no_balance=np.zeros(row_block*col_round,int)    #不考虑冲突
    ideal_conflict=np.zeros(row_block*col_round,int)    #考虑冲突的理想情况
    has_balance_conflict=np.zeros(row_block*col_round,int)  #考虑冲突的我们的性能
    no_balance_conflict=np.zeros(row_block*col_round,int)   #考虑冲突的实际情况

    PE_num=16
    for i in range(row_block):
        for j in range(col_round):
            ideal[i*col_round+j]=submat_nnz_num[i*col_round+j]/PE_num
            has_balance[i*col_round+j]=comp(submat_row_len[i*col_round+j],PE_num,1,0)
            no_balance[i*col_round+j]=comp(submat_row_len[i*col_round+j],PE_num,0,0)
            ideal_conflict[i*col_round+j]=max(max(submat_row_len[i*col_round+j])*4,ideal[i*col_round+j])
            has_balance_conflict[i*col_round+j]=comp(submat_row_len[i*col_round+j],PE_num,1,1)
            no_balance_conflict[i*col_round+j]=comp(submat_row_len[i*col_round+j],PE_num,0,1)
    
    return [ideal,has_balance,no_balance,ideal_conflict,has_balance_conflict,no_balance_conflict]

dataset_name_path='/home/wanghongyi22/SpMM/matrix_name0.txt'
dataset_name_file=open(dataset_name_path,'r')

dataset_all_path=dataset_name_file.readlines()
for i in range(len(dataset_all_path)):
    dataset_all_path[i]=dataset_all_path[i].rstrip()

balance_detail=open('/home/wanghongyi22/SpMM/result/balance_detail0.txt','w')
balance_summary=open('/home/wanghongyi22/SpMM/result/balance_summary0.txt','w')

for i in range(15,len(dataset_all_path)):       #  len(dataset_all_path)
    dataset_name=(dataset_all_path[i].split('/'))[-1]
    adata=sc.read(dataset_all_path[i])
    data=adata.X
    row=adata.shape[0]
    col=adata.shape[1]
    sparsity=data.nnz/row/col

    [ideal,has_balance,no_balance,ideal_conflict,has_balance_conflict,no_balance_conflict]=load_balance_sim(data,row,col)

    balance_summary.write(dataset_name+','+str(sum(ideal))+','+str(sum(has_balance))+','+str(sum(no_balance))+','+str(sum(ideal_conflict))+','+str(sum(has_balance_conflict))+','+str(sum(no_balance_conflict)))
    balance_summary.write(','+str(max(ideal))+','+str(max(has_balance))+','+str(max(no_balance))+','+str(max(ideal_conflict))+','+str(max(has_balance_conflict))+','+str(max(no_balance_conflict))+'\n')

    balance_detail.write(dataset_name+'\n')
    for j in range(len(ideal)):      #因为前两个是0
        balance_detail.write(str(ideal[j])+',')
    balance_detail.write('\n')
    for j in range(len(has_balance)):      #因为前两个是0
        balance_detail.write(str(has_balance[j])+',')
    balance_detail.write('\n')
    for j in range(len(no_balance)):      #因为前两个是0
        balance_detail.write(str(no_balance[j])+',')
    balance_detail.write('\n')
    for j in range(len(ideal_conflict)):      #因为前两个是0
        balance_detail.write(str(ideal_conflict[j])+',')
    balance_detail.write('\n')
    for j in range(len(has_balance_conflict)):      #因为前两个是0
        balance_detail.write(str(has_balance_conflict[j])+',')
    balance_detail.write('\n')
    for j in range(len(no_balance_conflict)):      #因为前两个是0
        balance_detail.write(str(no_balance_conflict[j])+',')
    balance_detail.write('\n')


balance_detail.close()
balance_summary.close()
