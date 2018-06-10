from __future__ import division
import numpy as np
import os
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

def get_Ragaclass(rec,fname):
    ragaclass='99'
    for tuples in rec:
                words2=tuples.split(",")
                if(words2[0]==fname):
                    ragaclass=words2[9]
                    return int(ragaclass)
                else:
                    ragaclass='99'
                    continue
    return int(ragaclass)
    

path=os.getcwd()
reverse_enc={'Sa': 1, 're': 2, 'Re': 3, 'ga':4, 'Ga':5, 'ma':6, 'Ma':7, 'Pa':8, 'dha':9, 'Dha':10, 'ni':11, 'Ni':12}
All_notes_seq=[]
All_filename=[]
All_ragaclass=[]
#with open("All_transcription.csv","w") as t_fname,open("metadata.csv","r") as f2:
f2=open("metadata.csv","r")
rec=f2.readlines()
for file1 in os.listdir(path):
    if file1.endswith(".txt"):
        listOfnotes=""
        enc_listOfnotes=[]
        f=open(file1,"r")
        lines=f.readlines()
        for line in lines:
            words=line.split(",")
            listOfnotes=listOfnotes+(words[2].rstrip("\n"))+" "
            note=words[2].rstrip("\n")
            enc_listOfnotes.append(reverse_enc[note])
        All_notes_seq.append(enc_listOfnotes)
            
        temp_fname_list=file1.split(".mp3")
        fname="./wave/"+temp_fname_list[0]+".wav"
        ragaclass=get_Ragaclass(rec,fname)
        if(ragaclass==99):
            continue
        All_filename.append(file1)
        All_ragaclass.append(int(ragaclass))


mylist = list(set(All_ragaclass))
print("All unique notes:",mylist)
padded_seq = pad_sequences(All_notes_seq, padding='post')
#print("length of padded_seq:",len(padded_seq))
#print("length of All_filename:",len(All_filename))
#print("length of All_ragaclass:",len(All_ragaclass))
data=[]
for t in range(len(All_filename)):
#    t1=(All_filename[t],)+(All_ragaclass[t],)    
    t1=(All_filename[t],)+(All_ragaclass[t],)+tuple(padded_seq[t])
    data.append(t1)
df = pd.DataFrame(data)
len1=len(df.columns)
fname="final_transcription"

cmd="sort -t',' -n -k"+str(len1)+","+str(len1)+" "+fname+".csv"
cmd1="mkdir training_transcript1"
os.system(cmd1)
cmd2="mkdir Data"
os.system(cmd2)

print(df.shape)
#print(cmd)
df.to_csv("./training_transcript1/"+"final_transcription.csv", encoding='utf-8', index=False)
#os.system(cmd)
##f=open("final_transcription.csv","w")

train_np=np.empty([0, 7285], dtype=object)
test_np=np.empty([0, 7285], dtype=object)
for x,y in df.groupby(1):
    len2=len(y)
    offset=int((3/4)*len2)
#    print("len:",len2,"offset:",offset)
    data1=y.values
    train,test = data1[:offset,:],data1[offset:,:]
    
#    print("length:",len(test))
    df1=pd.DataFrame(train)
    df2=pd.DataFrame(test)
    train_np=np.concatenate((train_np,train),axis=0)
    test_np=np.concatenate((test_np,test),axis=0)
    
#    print("train len:",len(df1),"test len:",len(df2))
    
    df1.to_csv("./training_transcript1/"+fname+"_"+str(x)+".csv", encoding='utf-8', index=False)
#    print("==========================================")
    

train_df=pd.DataFrame(train_np)    
train_df.to_csv("./Data/traindata.csv", encoding='utf-8', index=False)

train_df1=train_df.drop(train_df.columns[[0]], axis=1)
train_df1.to_csv("./Data/traindata_temp.csv", encoding='utf-8', index=False)

test_df=pd.DataFrame(test_np)    
test_df.to_csv("./Data/testdata.csv", encoding='utf-8', index=False)

test_df1=test_df.drop(test_df.columns[[0]], axis=1)
test_df1.to_csv("./Data/testdata_temp.csv", encoding='utf-8', index=False)

#for x in range(1,16):
#    df1=df[df[1].isin([x])]
#    print(df1[df1.columns[0:2]])
#    print("======================================================")
#    df1.to_csv(fname+"_"+str(x)+".csv", encoding='utf-8', index=False)
#        print(file1)
    