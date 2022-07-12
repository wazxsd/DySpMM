from math import ceil
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import time

class DySpMM(object):
    
    def __init__(self,data,A_row,A_col,B_col,bandwidthA,bandwidthB,balance_flag):
        self.data=data
        self.A_row=A_row
        self.A_col=A_col
        self.B_col=B_col
        self.sparsity=data.nnz/self.A_row/self.A_col

        self.bandwidthA=bandwidthA
        self.bandwidthB=bandwidthB
        self.bandwidthC=8*512
        self.sparse_element_width=48

        self.PEGnum=16
        self.PEnum=16

        self.M0=4096*24     #每个PEG中的缓存大小
        self.K0=1024
        self.N0=4
        #self.balance_flag=balance_flag
        
        return
    
    def Dynamic_partition(self):
        min_time=2**31
        # for i in [1,2,4,8,16,32,64]:
        #     for j in range(1,M0_ini):      #这个是最终每一个PE读的小子块的大小
        #         time=max(2*sparsity*min(j*int(PE_tile_num/i),row)/bandwidthA,min(N0*i,B_col)/bandwidthB)*ceil(row/(j*(int(PE_tile_num/i))))*ceil(B_col/(N0*i))

        for i in [1,2,4,8,16]:
            time_estimate=max(self.sparsity*min(self.M0*int(self.PEGnum/i),self.A_row)*self.sparse_element_width/self.bandwidthA,min(self.N0*i,self.B_col)*32/self.bandwidthB)*ceil(self.A_row/(self.M0*(int(self.PEGnum/i))))*ceil(self.B_col/(self.N0*i))
            if time_estimate<min_time:
                min_time=time_estimate
                self.N_blocknum=i
                self.M_blocknum=int(self.PEGnum/i)
        return

    def preprocessing(self):
        self.row_round=int((self.A_row-1)/ self.M0/self.M_blocknum)+1      #注意
        self.row_block=self.row_round*self.M_blocknum         #划分的小块数量  和round区分在于一次任务中处理多个小块
        self.col_round=int((self.A_col-1)/self.K0)+1
        self.B_col_round=int((self.B_col-1)/self.N0/self.N_blocknum)+1

        self.submat_row_len=[]           #表示每行有多少个非零元
        self.submat_nnz_num=[]
        for i in range(self.row_block*self.col_round):
            self.submat_row_len.append(np.zeros(self.M0,int))
            self.submat_nnz_num.append(0)

        for i in range(self.A_row):
            row_block_ite=int(i/(self.M0*self.M_blocknum))
            row_ite=row_block_ite*self.M_blocknum+i%(self.M0*self.M_blocknum)%self.M_blocknum
            rela_row=int((i%(self.M0*self.M_blocknum))/self.M_blocknum)
            for j in range(self.data.indptr[i],self.data.indptr[i+1]):
                col_ite=int(self.data.indices[j]/self.K0)
                rela_col=self.data.indices[j]%self.K0
                self.submat_row_len[row_ite*self.col_round+col_ite][rela_row]+=1
                self.submat_nnz_num[row_ite*self.col_round+col_ite]+=1
        return
    
    def readA_time(self):           #读A时间，需要确定读入方式后进一步完善
        self.readA_time_queue=[]
        for i in range(self.row_round):
            for j in range(self.col_round):
                readA_unbalanced_time=0     
                for k in range(self.M_blocknum):
                    sub_readA_time=ceil(self.submat_nnz_num[(i*self.M_blocknum+k)*self.col_round+j]*self.sparse_element_width/(self.bandwidthA/self.M_blocknum))
                    if readA_unbalanced_time<sub_readA_time:
                        readA_unbalanced_time=sub_readA_time
                self.readA_time_queue.append(ceil(readA_unbalanced_time)+2)     #+2信号时间
        return

    def readB_time(self):
        self.readB_time_queue=[]
        for i in range(self.row_round):
            for j in range(self.col_round):
                if j ==self.col_round-1:
                    readB_time=((self.A_col-1)%self.K0+1)*(self.N0*self.N_blocknum)*32/self.bandwidthB
                else:
                    readB_time=self.K0*(self.N0*self.N_blocknum)*32/self.bandwidthB
                self.readB_time_queue.append(ceil(readB_time)+2)
    
    def conflict_avoid(row_len):            
        latency=4   #9         #需要的延时
        color=np.zeros(latency,int)     #相当于涂色问题了
        for i in range(len(row_len)):
            if row_len[i]!=0:
                min_ind=np.argmin(color)
                color[min_ind]+=row_len[i]
        return max(color)*latency

    def PEG_comp(self,row_len,balance_flag,conflict_flag):
        task_queue=[]
        for i in range(self.PEnum):
            task_queue.append([])
        if balance_flag:
            task_lenth=np.zeros(self.PEnum,int)
            for i in range(len(row_len)):
                if row_len[i]!=0:
                    min_ind=np.argmin(task_lenth)
                    task_lenth[min_ind]+=row_len[i]
                    task_queue[min_ind].append(row_len[i])
        else:
            for i in range(len(row_len)):
                if row_len[i]!=0:
                    task_queue[i%self.PEnum].append(row_len[i])
        
        task_final_time=[]
        if conflict_flag:
            for i in range(self.PEnum):
                task_final_time.append(self.conflict_avoid(task_queue[i]))
        else:
            for i in range(self.PEnum):
                task_final_time.append(sum(task_queue[i]))
        
        return max(task_final_time)     #返回执行时间最长的PE

    def comp_time(self):
        self.ideal_comp_time_queue=[]
        self.balance_comp_time_queue=[]
        self.unbalance_comp_time_queue=[]
        if self.row>1000000:    #为了降低测试整体时间
            conflict_flag=0
        else:
            conflict_flag=1

        for i in range(self.row_round):
            for j in range(self.col_round):
                PEG_balance_maxtime=0           #不同PEG任务中中时间最长的块
                PEG_unbalance_maxtime=0         #不做平衡
                PEG_ideal_maxtime=0             #PEG内部最优平衡的时间
                #PEG_ideal_time=0                #不同PEG之间也平衡时的最优时间
                for k in range(self.M_blocknum):
                    PEG_balance_time=self.PEG_comp(self.submat_row_len[(i*self.M_blocknum+k)*self.col_round+j],1,conflict_flag)
                    PEG_unbalance_time=self.PEG_comp(self.submat_row_len[(i*self.M_blocknum+k)*self.col_round+j],0,conflict_flag)
                    PEG_ideal_time=max(max(self.submat_row_len[(i*self.M_blocknum+k)*self.col_round+j])*4,self.submat_nnz_num[(i*self.M_blocknum+k)*self.col_round+j]/self.PEnum)
                    #PEG的理想时间是只执行最长行冲突避免和所有负载均衡后的最大值
                    if PEG_balance_maxtime<PEG_balance_time:
                        PEG_balance_maxtime=PEG_balance_time
                    if PEG_unbalance_maxtime<PEG_unbalance_time:
                        PEG_unbalance_maxtime=PEG_unbalance_time
                    if PEG_ideal_maxtime<PEG_ideal_time:
                        PEG_ideal_maxtime=PEG_ideal_time
                
                self.ideal_comp_time_queue.append(ceil(PEG_ideal_maxtime)+15+2)     #最后一个元素算完15个周期+2周期信号
                self.balance_comp_time_queue.append(ceil(PEG_balance_maxtime)+15+2)
                self.unbalance_comp_time_queue.append(ceil(PEG_unbalance_maxtime+15+2))
        return
                
    def writeC_time(self):
        self.WB_time_queue=[]
        for i in range(self.row_round):
            if i ==self.row_round-1:
                WB_time=((self.A_row-1)%(self.M0*self.M_blocknum)+1)*(self.N0*self.N_blocknum)*32/self.bandwidthC
            else:
                WB_time=(self.M0*self.M_blocknum)*(self.N0*self.N_blocknum)*32/self.bandwidthC
            self.WB_time_queue.append(ceil(WB_time)+2)
        return

    def pipeline(self,readA_time_queue,readB_time_queue,comp_time_queue,WB_time_queue):
        readA_timeline=[[0,0],[0,0]]
        readB_timeline=[[0,0],[0,0]]        #0填充一下
        comp_timeline=[[0,0],[0,0]]
        WB_timeline=[[0,0],[0,0]]      #分别代表三个子任务的每次开始及结束时间
        for i in range(self.B_col_round):      
            for j in range(self.row_round):
                for k in range(self.col_round):
                    now_queue_index=j*self.col_round+k       #由于换B的列对计算模式不会产生影响  可以复用
                    now_ite=i*self.row_round*self.col_round+j*self.col_round+k
                    A_start=max(readA_timeline[now_ite-1+2][1],comp_timeline[now_ite-2+2][1])
                    readA_timeline.append([A_start,A_start+readA_time_queue[now_queue_index]])
                    # plt.plot([A_start,A_start+readA_time_queue[now_ite]],[4,4],color='red')
                    B_start=max(readB_timeline[now_ite-1+2][1],comp_timeline[now_ite-2+2][1])        #前一个读完且前两个的B算完
                    readB_timeline.append([B_start,B_start+readB_time_queue[now_queue_index]])
                    # plt.plot([B_start,B_start+readB_time_queue[now_ite]],[3,3],color='green')
                    if k==0:

                        comp_start=max(readA_timeline[now_ite+2][1],readB_timeline[now_ite+2][1],WB_timeline[-1][1])
                        comp_timeline.append([comp_start,comp_start+ comp_time_queue[now_queue_index]])
                    else:
                        comp_start=max(readA_timeline[now_ite+2][1],readB_timeline[now_ite+2][1],comp_timeline[now_ite-1+2][1])          #同一个读完且前一个算完
                        comp_timeline.append([comp_start,comp_start+comp_time_queue[now_queue_index]])

                WB_start=comp_timeline[-1][1]
                WB_timeline.append([WB_start,WB_start+WB_time_queue[j]])

        return [readA_timeline[2:],readB_timeline[2:],comp_timeline[2:],WB_timeline[2:]]

    def metric(self):
        self.total_readAtime=sum(self.readA_time_queue)*self.B_col_round
        self.total_readBtime=sum(self.readB_time_queue)*self.B_col_round
        self.total_balance_comp_time=sum(self.balance_comp_time_queue)*self.B_col_round
        self.total_unbalance_comp_time=sum(self.unbalance_comp_time_queue)*self.B_col_round
        self.total_ideal_comp_time=sum(self.ideal_comp_time_queue)*self.B_col_round
        self.total_WBtime=sum(self.WB_time_queue)*self.B_col_round
        self.balance_perf=self.balance_WB_timeline[-1][1]
        self.unbalance_perf=self.unbalance_WB_timeline[-1][1]
        self.ideal_perf=self.ideal_WB_timeline[-1][1]
        return

    def run(self):
        self.Dynamic_partition()
        self.preprocessing()
        self.readA_time()
        self.readB_time()
        self.comp_time()
        self.writeC_time()
        [self.balance_readA_timeline,self.balance_readB_timeline,self.balance_comp_timeline,self.balance_WB_timeline]=self.pipeline(self.readA_time_queue,self.readB_time_queue,self.balance_comp_time_queue,self.WB_time_queue)
        [self.unbalance_readA_timeline,self.unbalance_readB_timeline,self.unbalance_comp_timeline,self.unbalance_WB_timeline]=self.pipeline(self.readA_time_queue,self.readB_time_queue,self.unbalance_comp_time_queue,self.WB_time_queue)
        [self.ideal_readA_timeline,self.ideal_readB_timeline,self.ideal_comp_timeline,self.ideal_WB_timeline]=self.pipeline(self.readA_time_queue,self.readB_time_queue,self.ideal_comp_time_queue,self.WB_time_queue)
        self.metric()

    #return [M_blocknum,N_blocknum,B_col_round*sum(readA_time_queue),B_col_round*sum(readB_time_queue),B_col_round*sum(comp_time_queue),B_col_round*sum(WB_time_queue),WB_timeline[-1][1],readA_timeline[2:row_round*col_round+2],readB_timeline[2:row_round*col_round+2],comp_timeline[2:row_round*col_round+2],WB_timeline[2:row_round+2]]


####  测试数据路径读入
dataset_name_path='/home/wanghongyi22/SpMM/matrix_name0.txt'
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

############  DySpMM  测试

DySpMM_file=open('/home/wanghongyi22/SpMM/result/DySpMM_perf_summary.txt','w')
DySpMM_balance_timeline_file=open('/home/wanghongyi22/SpMM/result/DySpMM_balance_timeline.txt','w')
DySpMM_unbalance_timeline_file=open('/home/wanghongyi22/SpMM/result/DySpMM_nobalance_timeline.txt','w')
DySpMM_ideal_timeline_file=open('/home/wanghongyi22/SpMM/result/DySpMM_ideal_timeline.txt','w')

def timeline_file_write(file_ptr,readA_timeline,readB_timeline,comp_timeline,WB_timeline):
    for j in range(len(readA_timeline)):      #因为前两个是0
        file_ptr.write(str(readA_timeline[j][0])+','+str(readA_timeline[j][1])+',')
    file_ptr.write('\n')
    for j in range(len(readB_timeline)):      #因为前两个是0
        file_ptr.write(str(readB_timeline[j][0])+','+str(readB_timeline[j][1])+',')
    file_ptr.write('\n')
    for j in range(len(comp_timeline)):      #因为前两个是0
        file_ptr.write(str(comp_timeline[j][0])+','+str(comp_timeline[j][1])+',')
    file_ptr.write('\n')
    for j in range(len(WB_timeline)):      #因为前两个是0
        file_ptr.write(str(WB_timeline[j][0])+','+str(WB_timeline[j][1])+',')
    file_ptr.write('\n')
    return

for i in range(len(dataset_all_path)):       #  len(dataset_all_path)
    dataset_name=(dataset_all_path[i].split('/'))[-1]
    adata=sc.read(dataset_all_path[i])
    data=adata.X
    row=adata.shape[0]
    col=adata.shape[1]
    sparsity=data.nnz/row/col
    bandwidthA=512*8
    bandwidthB=512*4

    DySpMM_file.write(dataset_name+','+str(row)+','+str(col)+','+str(data.nnz)+','+str(sparsity)+',')
    # DySpMM_balance_timeline_file.write(dataset_name+','+str(row)+','+str(col)+','+str(data.nnz)+','+str(sparsity)+',')
    # DySpMM_unbalance_timeline_file.write(dataset_name+','+str(row)+','+str(col)+','+str(data.nnz)+','+str(sparsity)+',')
    # DySpMM_ideal_timeline_file.write(dataset_name+','+str(row)+','+str(col)+','+str(data.nnz)+','+str(sparsity)+',')
    for B_col in [8,16,32,64,128,256,512]:
        Acc=DySpMM(data,row,col,B_col,bandwidthA,bandwidthB,0)      #Balanceflag暂时没使用
        Acc.run()

        DySpMM_file.write(str(Acc.M_blocknum)+','+str(Acc.N_blocknum)+',')
        DySpMM_file.write(str(Acc.total_readAtime)+','+str(Acc.total_readBtime)+',')
        DySpMM_file.write(str(Acc.total_ideal_comp_time)+','+str(Acc.total_balance_comp_time)+','+str(Acc.total_unbalance_comp_time)+',')
        DySpMM_file.write(str(Acc.total_WBtime)+','+str(Acc.ideal_perf)+','+str(Acc.balance_perf)+','+str(Acc.unbalance_perf)+',')

        DySpMM_balance_timeline_file.write(dataset_name+',N='+str(B_col)+'\n')
        timeline_file_write(DySpMM_balance_timeline_file,Acc.balance_readA_timeline,Acc.balance_readB_timeline,Acc.balance_comp_timeline,Acc.balance_WB_timeline)
        DySpMM_unbalance_timeline_file.write(dataset_name+',N='+str(B_col)+'\n')
        timeline_file_write(DySpMM_unbalance_timeline_file,Acc.unbalance_readA_timeline,Acc.unbalance_readB_timeline,Acc.unbalance_comp_timeline,Acc.unbalance_WB_timeline)
        DySpMM_ideal_timeline_file.write(dataset_name+',N='+str(B_col)+'\n')
        timeline_file_write(DySpMM_ideal_timeline_file,Acc.ideal_readA_timeline,Acc.ideal_readB_timeline,Acc.ideal_comp_timeline,Acc.ideal_WB_timeline)
        print(dataset_name+str(B_col))

    DySpMM_file.write('\n')

DySpMM_file.close()
DySpMM_balance_timeline_file.close()
DySpMM_unbalance_timeline_file.close()
DySpMM_ideal_timeline_file.close()

##    new_arch DSE使用

# for k in [2,4,6,8,10]:
#     result_txt=".\\3PE_result\\"+str(k)+"_512sim_result_more1.txt"
#     pure_num=open(result_txt,'w')
    
#     for i in range(len(testbench)):
#         dataset_name=testbench[i]
#         adata=sc.read(r"C:\Users\10784\Desktop\simu\\"+dataset_name+"\\"+dataset_name+".mtx")
#         data=adata.X
#         row=adata.shape[0]
#         col=adata.shape[1]
#         sparsity=data.nnz/row/col
#         if row>500000:
#             conflict_flag=0
#         else:
#             conflict_flag=1
#         print(dataset_name+':    '+str(row)+'    '+str(col)+'    '+str(data.nnz)+'    '+str(sparsity))
#         for B_col in [8,16,32,64,128]:
#             new_arch_time=new_arch_sim(data,row,col,B_col,conflict_flag,k*512,(12-k)*512)
#             pure_num.write(str(new_arch_time)+'  ')
#             print(str(new_arch_time)+'  ')
#         pure_num.write('\n')

#     pure_num.close()