# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import matplotlib.pyplot as plt
from collections import defaultdict
from math import log
import os
import sys
import Queue

def process(ftonic,fname):
#    ftonic = 256
#    fname = sys.argv[1]
    pitch_file = open(fname, "r")
    bin_file = open("bin.csv", "w")
    lines =  pitch_file.read().split('\n')
    list1=[]
    d = defaultdict(list)
    for i in range(len(lines)-1):
        list1.append(lines[i].split(','))
#        print("list:",list1)
        
        if(float(list1[i][1])!=0):
#            print("here",list1[i][1])
#            print 'i: {l} and list1[i][1]: {m}'.format(l=i,m= list1[i][1])
            n = round(1200*(log(float(list1[i][1])/ftonic)/log(2)),0)%1200
            d[n].append(float(list1[i][1]))
            rec=""
            rec=list1[i][0]
            rec+=','+list1[i][1]
            rec+=','+str(n)+"\n"
            bin_file.write(rec)
    bin_file.close()
    pitch_file.close()

def reverseQueue(my_queue):
#    aux_stack = Stack()
    aux_stack = Queue.LifoQueue(maxsize=3)
    while not my_queue.empty():
        aux_stack.put(my_queue.get())
        
    while not aux_stack.empty():
        my_queue.put(aux_stack.get())
    return my_queue
def getCandidate(bin_num,delta):
#==============================================================================
#     encoding of notes
#     Sa - 0 - 1 or 2  - 0 or 1200
#     re - 1 - 256/243 - 111.7312
#     Re - 2 - 9/8 - 203.91
#     ga - 3 - 32/27 - 315.64
#     Ga - 4 - 5/4 - 386.31
#     ma - 5 - 4/3 - 498.04
#     Ma - 6 - 45/32 - 609.77
#     Pa - 7 - 3/2 - 701.955
#     dha - 8 - 128/81 - 813.68
#     Dha - 9 - 5/3 - 884.35
#     ni - 10 - 16/9 - 996.08
#     Ni - 11 - 15/8 - 1088.26
#==============================================================================

     candidate_note=12
     if((bin_num>=0 and bin_num<=30) or((bin_num>=1171 and bin_num<=1200))):
         candidate_note=0
     elif((bin_num>=111 and bin_num<=142) or (bin_num>=81 and bin_num<111)):
         candidate_note=1
     elif((bin_num>=203 and bin_num<=234) or (bin_num>=170 and bin_num<203)):     
         candidate_note=2
     elif((bin_num>=315 and bin_num<=346) or (bin_num>=285 and bin_num<315)):
         candidate_note=3
     elif((bin_num>=386 and bin_num<=417) or (bin_num>=365 and bin_num<386)):
         candidate_note=4
     elif((bin_num>=498 and bin_num<=529) or (bin_num>=468 and bin_num<498)):
         candidate_note=5
     elif((bin_num>=609 and bin_num<=640) or (bin_num>=779 and bin_num<609)):
         candidate_note=6
     elif((bin_num>=701 and bin_num<=732) or (bin_num>=671 and bin_num<701)):
         candidate_note=7
     elif((bin_num>=813 and bin_num<=844) or (bin_num>=883 and bin_num<813)):
         candidate_note=8
     elif((bin_num>=884 and bin_num<=915) or (bin_num>=854 and bin_num<884)):
         candidate_note=9
     elif((bin_num>=996 and bin_num<=1027) or (bin_num>=996 and bin_num<996)):
         candidate_note=10
     elif((bin_num>=1088 and bin_num<=1119) or (bin_num>=1058 and bin_num<1088)):
         candidate_note=11
     else:
         candidate_note=12
     return candidate_note
  

#==============================================================================
# 
#     candidate_note=12
#     if((bin_num>=0 and bin_num<=0+delta) or((bin_num>=1200-delta and bin_num<=1200))):
#         candidate_note=0
#     elif((bin_num>=111 and bin_num<=142) or (bin_num>=81 and bin_num<111)):
#         candidate_note=1
#     elif((bin_num>=203 and bin_num<=234) or (bin_num>=170 and bin_num<203)):     
#         candidate_note=2
#     elif((bin_num>=315 and bin_num<=346) or (bin_num>=285 and bin_num<315)):
#         candidate_note=3
#     elif((bin_num>=386 and bin_num<=417) or (bin_num>=365 and bin_num<386)):
#         candidate_note=4
#     elif((bin_num>=498 and bin_num<=529) or (bin_num>=468 and bin_num<498)):
#         candidate_note=5
#     elif((bin_num>=609 and bin_num<=640) or (bin_num>=779 and bin_num<609)):
#         candidate_note=6
#     elif((bin_num>=701 and bin_num<=732) or (bin_num>=671 and bin_num<701)):
#         candidate_note=7
#     elif((bin_num>=813 and bin_num<=844) or (bin_num>=883 and bin_num<813)):
#         candidate_note=8
#     elif((bin_num>=884 and bin_num<=915) or (bin_num>=854 and bin_num<884)):
#         candidate_note=9
#     elif((bin_num>=996 and bin_num<=1027) or (bin_num>=996 and bin_num<996)):
#         candidate_note=10
#     elif((bin_num>=1088 and bin_num<=1119) or (bin_num>=1058 and bin_num<1088)):
#         candidate_note=11
#     else:
#         candidate_note=12
# 
#==============================================================================


   

def transcriptNotes():
    #Melody Tolerance
    delta=30.5
    #Threshold Duration
    T_d=0.02
    err=0
    start=0
    seq_bin=[]
    curr_candidate=0
    candidate_note=0
    bin_rfile = open("bin.csv", "r")
    record = bin_rfile.read().split('\n')
    transcription=[]
    records=[]
#    q=[]
    p = [x.split(',') for x in record]
    k=1;
    for j in p:
        k+=1
        try:
            records.append(map(float,j))
        except ValueError:
#            print 'Line {i} is corrupt!'.format(i = k)
            break
    temp_index=0
    #flag1 is set to jump back the index to temp_index
    #set by either last 3 records or when 3 consecutive error occur 
    flag1=0
    index =0    
    while(index <=(len(records)-1)):
        if(flag1==1):
            index=temp_index
            flag1=0
            
        time=records[index][0]
        bin_num=records[index][2]
        candidate_note=getCandidate(bin_num,delta)
        if(index==start):
#            print("start:",start)
        
#            flag is set when last 3 record contain a curr_candidate record
#            therfore update start
            flag=0
            err=0
            q = Queue.Queue(maxsize=3)
            with q.mutex:
                q.queue.clear()
            if(candidate_note==12):
                start=index+1
                index+=1
                continue
            else:
                seq_bin[:] = []
                seq_bin.append([time,bin_num])
                curr_candidate=candidate_note
                q.put([index,bin_num])
        else:
            if(candidate_note != 12):
                if(q.full()):
                    last=q.get()
                    q.put([index,bin_num,candidate_note,time])
                else:
                    q.put([index,bin_num,candidate_note,time])
                
            if(candidate_note==curr_candidate):
                seq_bin.append([time,bin_num])
            else:
                if(candidate_note==12):
                    index+=1
                    continue
#                print 'current: {i} and candidate: {j}'.format(i=curr_candidate,j= candidate_note)
                err+=1
                qtuple=[]
                if(err==3):
                    err=0
                    q=reverseQueue(q)
                    while(not(q.empty())):
                        flag=0
                        qtuple=q.get()
#                        print(qtuple)
                        candidate_note=getCandidate(qtuple[1],delta)
                        if(candidate_note==curr_candidate):
                            if(not(q.empty())):
#                                print("here1")
                                qtuple=q.get()
                                start=qtuple[0]
                                temp_index=qtuple[0]
                                flag=1
                                flag1=1
                                start_time=seq_bin[0][0]
                                end_time=qtuple[3]
#                                seq_bin[:] = []
                                if((end_time-start_time)>=T_d):
                                    transcription.append([start_time,end_time,curr_candidate])
                                break
                            else:
#                                print("here2")
                                continue
                        else:
#                            print("here4",flag)
                            continue
#                    print("=========================================")
                    if(q.empty() and flag==0):
                        start=qtuple[0]
                        temp_index=qtuple[0]
                        flag1=1
#                        print("Index:",index)
                        start_time=seq_bin[0][0]
                        end_time=seq_bin[len(seq_bin)-1][0]
                        index=start
#                        seq_bin[:] = []
#                        print 'start: {i} and qtuple: {j}'.format(i=start,j= qtuple)
                        if((end_time-start_time)>=T_d):
                                transcription.append([start_time,end_time,curr_candidate])
                        continue
        index+=1
                        
#                        print("here3",start)
#    print(transcription)
    return transcription
                    
                           
def getTonic(fname):
    with open("metadata.csv","r") as f:
        rec=f.readlines()
        rec=[x.strip() for x in rec]
        if(fname in rec[0]):
            ftonic=rec[4]
            return float(ftonic)
        else:
            ftonic = '256'
    return float(ftonic)
               
    
if ( __name__ == "__main__"):
    fname = sys.argv[1]
#    ftonic = 256
    temp_fname=fname.find(".csv")
    transcript_fname=fname[:temp_fname]+"_transcript.csv"
    print("FileName:",fname)
    tonicf=getTonic(fname)
    print("Tonic Frequency",tonicf)    
    process(tonicf,fname)
    transcript=[]
    transcript=transcriptNotes()
    
#==============================================================================
#     encoding of notes
#     Sa - 0 - 1 or 2  - 0 or 1200
#     re - 1 - 256/243 - 111.7312
#     Re - 2 - 9/8 - 203.91
#     ga - 3 - 32/27 - 315.64
#     Ga - 4 - 5/4 - 386.31
#     ma - 5 - 4/3 - 498.04
#     Ma - 6 - 45/32 - 609.77
#     Pa - 7 - 3/2 - 701.955
#     dha - 8 - 128/81 - 813.68
#     Dha - 9 - 5/3 - 884.35
#     ni - 10 - 16/9 - 996.08
#     Ni - 11 - 15/8 - 1088.26
#==============================================================================
    encoding=['Sa','re','Re','ga','Ga','ma','Ma','Pa','dha','Dha','ni','Ni']
    reverse_enc={'Sa': 0, 're': 1, 'Re': 2, 'ga':3, 'Ga':4, 'ma':5, 'Ma':6, 'Pa':7, 'dha':8, 'Dha':9, 'ni':10, 'Ni':11}
    t_fname=open(transcript_fname,"w")
    for x in transcript:
        t_fname.write(str(x[0])+","+str(x[1])+","+encoding[x[2]]+"\n")
#    print(transcript)
