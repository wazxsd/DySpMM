import numpy as np
import scanpy as sc
import math

dataset_name_path='/home/wanghongyi22/SpMM/matrix_name1.txt'
dataset_name_file=open(dataset_name_path,'r')

dataset_name=dataset_name_file.readlines()
for i in range(len(dataset_name)):
    dataset_name[i]=dataset_name[i].rstrip()
# print(dataset_name[0:3])

def number_width(num):      #判断一个数字的位数，统计数量级用
    if num>=1:
        count=0
        while num>10:
            num=num/10
            count+=1
    else:
        count=0
        while num<1:
            num=num*10
            count-=1
    return count

def Matrix_partition_sim(nnz,row,col,B_col):     #给出不同分块大小下的访存量大小  没有实际读数据集
    sparsity=nnz/row/col

    Memory_bound=12288*64*8     #以U280的缓存大小为限制

    #sextans的固定策略
    M0_fixed=12288*64
    N0_fixed=8

    #无约束  也即每个乘累加单元都能自由分配任务  parallel_mul=1
    N0_best=round(math.sqrt(Memory_bound*sparsity))
    M0_best=math.floor(Memory_bound/N0_best)
    if N0_best>B_col:
        N0_best=B_col
        M0_best=math.floor(Memory_bound/N0_best)
        if M0_best>row:
            M0_best=row
    elif M0_best>row:
        M0_best=row
        N0_best=math.floor(Memory_bound/M0_best)
        if N0_best>B_col:
            N0_best=B_col

    # #有一定并行度要求的时候  也即SIMD的并行度
    # Parallel_Mul=8          #指一个计算单元内并行的的乘法单元个数，决定了N0的最小因数
    # N0_ours=round(math.sqrt(Memory_bound*sparsity)/Parallel_Mul)*Parallel_Mul
    # M0_ours=math.floor(Memory_bound/N0_ours)
    # if N0_ours>B_col:
    #     N0_ours=math.ceil(B_col/Parallel_Mul)*Parallel_Mul
    #     M0_ours=math.floor(Memory_bound/N0_ours)
    #     if M0_ours>row:
    #         M0_ours=row
    # elif M0_ours>row:
    #     M0_ours=row
    #     N0_ours=math.floor(Memory_bound/M0_ours/Parallel_Mul)*Parallel_Mul
    #     if N0_ours>B_col:
    #         N0_ours=B_col

    #遍历我们的情况
    N0_ours=4
    M0_ours=Memory_bound/4
    Data_amount=(math.ceil(B_col/N0_ours)*nnz+math.ceil(row/M0_ours)*col*B_col)/8/1024      #拿4做初始值来当基准量
    for N0_test in [4,8,16,32,64,128]:  #最低的并行度是自己设定的
        M0_test=Memory_bound/N0_test
        if Data_amount>(math.ceil(B_col/N0_test)*nnz+math.ceil(row/M0_test)*col*B_col)/8/1024:
            Data_amount=(math.ceil(B_col/N0_test)*nnz+math.ceil(row/M0_test)*col*B_col)/8/1024
            M0_ours=M0_test
            N0_ours=N0_test

    Data_amount_sextans=(math.ceil(B_col/N0_fixed)*nnz+math.ceil(row/M0_fixed)*col*B_col)/8/1024        
    Data_amount_best=(math.ceil(B_col/N0_best)*nnz+math.ceil(row/M0_best)*col*B_col)/8/1024  
    Data_amount_ours=(math.ceil(B_col/N0_ours)*nnz+math.ceil(row/M0_ours)*col*B_col)/8/1024     #单位KB        

    return [Data_amount_sextans,M0_ours,N0_ours,Data_amount_ours,M0_best,N0_best,Data_amount_best]

row_sta=np.zeros(20,int)
nnz_sta=np.zeros(20,int)
sparsity_sta=np.zeros(20,int)

matrix_data_file=open('/home/wanghongyi22/SpMM/result/matrix_data_amount1.txt','w')


for i in range(len(dataset_name)):
# for i in range(3):  #先只选30个测一下访存数据量下降比例
    matrix=sc.read(dataset_name[i])
    matrix_data=matrix.X
    row=matrix_data.shape[0]
    col=matrix_data.shape[1]
    nnz=matrix_data.nnz
    sparsity=nnz/row/col

    row_sta[number_width(row)]+=1
    nnz_sta[number_width(nnz)]+=1
    sparsity_sta[abs(number_width(sparsity))]+=1

    # matrix_data_file.write((dataset_name[i].split('/'))[-1]+','+str(row)+','+str(col)+','+str(nnz)+','+str(sparsity)+',')

    # for j in [4,8,16,32,64,128,256,512]:
    #     [Data_amount_sextans,M0_ours,N0_ours,Data_amount_ours,M0_best,N0_best,Data_amount_best]=Matrix_partition_sim(nnz,row,col,j)
    #     matrix_data_file.write(str(Data_amount_sextans)+','+str(M0_ours)+','+str(N0_ours)+','+str(Data_amount_ours)+','+str(M0_best)+','+str(N0_best)+','+str(Data_amount_best)+',')
    # matrix_data_file.write('\n')


print(row_sta)
print(nnz_sta)
print(sparsity_sta)