from math import ceil
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

def sextans_sim(data,row,col,B_col,conflict_flag):
    M_blocknum=64
    N_blocknum=1

    M0=12288
    K0=2048
    N0=8

    bandwidthA=512*8        #512*8
    bandwidthB=512*4
    bandwidthC=512*8

    PE_num=64           #先不改  看看block和负载均衡的影响

    row_round=int((row-1)/M0/M_blocknum)+1
    row_block=row_round*M_blocknum         #划分的小块数量  和round区分在于PE并行处理不同小块
    col_round=int((col-1)/K0)+1
    B_col_round=int((B_col-1)/N0/N_blocknum)+1
    sparsity=data.nnz/row/col

    submat_row_len=[]           #表示每行有多少个非零元
    # submat_nnz_num=[]
    for i in range(row_block*col_round):
        submat_row_len.append(np.zeros(M0,int))
        # submat_nnz_num.append(0)

    for i in range(row):
        row_block_ite=int(i/(M0*M_blocknum))
        row_ite=row_block_ite*M_blocknum+i%(M0*M_blocknum)%M_blocknum
        rela_row=int((i%(M0*M_blocknum))/M_blocknum)
        for j in range(data.indptr[i],data.indptr[i+1]):
            col_ite=int(data.indices[j]/K0)
            rela_col=data.indices[j]%K0
            submat_row_len[row_ite*col_round+col_ite][rela_row]+=1
            # submat_nnz_num[row_ite*col_round+col_ite]+=1
    
    ca_lenths=np.zeros(row_block*col_round,int)
    if conflict_flag:
        for i in range(row_block):
            for j in range(col_round):
                ca_lenths[i*col_round+j]=conflict_avoid(submat_row_len[i*col_round+j])
    else:
        for i in range(row_block):
            for j in range(col_round):
                ca_lenths[i*col_round+j]=sum(submat_row_len[i*col_round+j])

    
    readA_time_queue=[]
    readB_time_queue=[]
    WB_time_queue=[]
    for t in range(B_col_round):          #原仿真  反复迭代B的列  但是对与我们实验中的方式好像不会出现多的有用数据  因为读B时间一样，如果改成B0=4则需要适配了
        for i in range(row_round):
            for j in range(col_round):
                if t==B_col_round-1:
                    if j ==col_round-1:
                        readB_time=((col-1)%K0+1)*((B_col-1)%(N0*N_blocknum)+1)*32/bandwidthB
                    else:
                        readB_time=K0*((B_col-1)%(N0*N_blocknum)+1)*32/bandwidthB
                else:
                    if j ==col_round-1:
                        readB_time=((col-1)%K0+1)*(N0*N_blocknum)*32/bandwidthB
                    else:
                        readB_time=K0*(N0*N_blocknum)*32/bandwidthB
                readB_time_queue.append(ceil(readB_time)+2)     #每个阶段的固定偏移
                readA_time=0
                for k in range(M_blocknum):
                    sub_readA_time=ca_lenths[(i*M_blocknum+k)*col_round+j]
                    if sub_readA_time>readA_time:
                        readA_time=sub_readA_time
                readA_time_queue.append(readA_time+2+12)        #固定偏移2+执行周期取12
            if t==B_col_round-1:
                if i ==row_round-1:
                    WB_time=((row-1)%(M0*M_blocknum)+1)*((B_col-1)%(N0*N_blocknum)+1)*32/bandwidthC
                else:
                    WB_time=(M0*M_blocknum)*((B_col-1)%(N0*N_blocknum)+1)*32/bandwidthC
            else:
                if i ==row_round-1:
                    WB_time=((row-1)%(M0*M_blocknum)+1)*(N0*N_blocknum)*32/bandwidthC
                else:
                    WB_time=(M0*M_blocknum)*(N0*N_blocknum)*32/bandwidthC
            WB_time_queue.append(ceil(WB_time)+2)

    readB_timeline=[[0,0],[0,0]]        #0填充一下
    comp_timeline=[[0,0],[0,0]]
    WB_timeline=[[0,0],[0,0]]    #分别代表三个子任务的每次开始及结束时间
    for i in range(B_col_round):
        for j in range(row_round):
            for k in range(col_round):
                now_ite=i*row_round*col_round+j*col_round+k
                B_start=max(readB_timeline[now_ite-1+2][1],comp_timeline[now_ite-2+2][1])        #前一个读完且前两个的B算完
                readB_timeline.append([B_start,B_start+readB_time_queue[now_ite]])
                # plt.plot([B_start,B_start+readB_time[now_ite]],[3,3],color='green')
                if k==0:
                    comp_start=max(readB_timeline[now_ite+2][1],WB_timeline[-1][1])
                    # comp_start=max(readB_timeline[now_ite+2][1],WB_timeline[-2][1])          #前两个写完
                    comp_timeline.append([comp_start,comp_start+readA_time_queue[now_ite]])
                else:
                    comp_start=max(readB_timeline[now_ite+2][1],comp_timeline[now_ite-1+2][1])          #同一个读完且前一个算完
                    comp_timeline.append([comp_start,comp_start+readA_time_queue[now_ite]])
                # plt.plot(comp_timeline[-1],[2,2],color='red')
            # WB_start=max(comp_timeline[-1][1],WB_timeline[-1][1])
            WB_start=comp_timeline[-1][1]
            WB_timeline.append([WB_start,WB_start+WB_time_queue[i*row_round+j]])
            # plt.plot(WB_timeline[-1],[1,1],color='blue')
    # plt.show()
    return [sum(readB_time_queue),sum(readA_time_queue),sum(WB_time_queue),WB_timeline[-1][1],readB_timeline[2:row_round*col_round+2],comp_timeline[2:row_round*col_round+2],WB_timeline[2:row_round+2]]
    
def conflict_avoid(row_len):
    latency=4   #9         #需要的延时
    color=np.zeros(latency,int)     #相当于涂色问题了
    for i in range(len(row_len)):
        if row_len[i]!=0:
            min_ind=np.argmin(color)
            color[min_ind]+=row_len[i]
    return max(color)*latency


dataset_name_path='/home/wanghongyi22/SpMM/matrix_name1.txt'
dataset_name_file=open(dataset_name_path,'r')

dataset_all_path=dataset_name_file.readlines()
for i in range(len(dataset_all_path)):
    dataset_all_path[i]=dataset_all_path[i].rstrip()

#####    test用
# dataset_name=testbench[0]
# adata=sc.read(r"C:\Users\10784\Desktop\Simulator\\"+dataset_name+"\\"+dataset_name+".mtx")
# data=adata.X
# row=adata.shape[0]
# col=adata.shape[1]
# B_col=8
# conflict_flag=1
# print(sextans_sim(data,row,col,B_col,conflict_flag))
# print(new_arch_sim(data,row,col,B_col,conflict_flag))

#######  sextans 测试 

sextans_file=open('/home/wanghongyi22/SpMM/result/sextans_perf1.txt','w')    #sextans_perf是最初的  并行度为8的测试结果文件
sextans_timeline_file=open('/home/wanghongyi22/SpMM/result/sextans_timeline1.txt','w')

for i in range(len(dataset_all_path)):       #  len(dataset_all_path)
    dataset_name=(dataset_all_path[i].split('/'))[-1]
    adata=sc.read(dataset_all_path[i])
    data=adata.X
    row=adata.shape[0]
    col=adata.shape[1]
    sparsity=data.nnz/row/col
    if row>1000000:
        conflict_flag=0
    else:
        conflict_flag=1
    sextans_file.write(dataset_name+','+str(row)+','+str(col)+','+str(data.nnz)+','+str(sparsity)+',')
    for B_col in [8,16,32,64,128,256,512]:
        [readB_time,comp_time,WB_time,all_time,readB_timeline,comp_timeline,WB_timeline]=sextans_sim(data,row,col,B_col,conflict_flag)
        sextans_file.write(str(readB_time)+','+str(comp_time)+','+str(WB_time)+','+str(all_time)+',')

        sextans_timeline_file.write(dataset_name+',N='+str(B_col)+'\n')
        for j in range(len(readB_timeline)):      
            sextans_timeline_file.write(str(readB_timeline[j][0])+','+str(readB_timeline[j][1])+',')
        sextans_timeline_file.write('\n')
        for j in range(len(comp_timeline)):      
            sextans_timeline_file.write(str(comp_timeline[j][0])+','+str(comp_timeline[j][1])+',')
        sextans_timeline_file.write('\n')
        for j in range(len(WB_timeline)):      
            sextans_timeline_file.write(str(WB_timeline[j][0])+','+str(WB_timeline[j][1])+',')
        sextans_timeline_file.write('\n')

    sextans_file.write('\n')

sextans_file.close()
sextans_timeline_file.close()
