import sys
import tkinter as tk
from collections import defaultdict
from math import log
import math
import matplotlib.pyplot as plt
import os
import numpy as np

class Example(tk.Frame):
    
    def __init__(self, root):

        tk.Frame.__init__(self, root)        
        self.canvas = tk.Canvas(root, borderwidth=0, background="#ffffff")
        self.frame = tk.Frame(self.canvas, background="#ffffff")
        self.vsb = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((4,4), window=self.frame, anchor="nw", 
                                  tags="self.frame")

        self.frame.bind("<Configure>", self.onFrameConfigure)
        self.process(256)
        self.PCD()
        #self.populate()

    def populate(self):
        global mtext
        self.mtext = tk.Text(self.frame,width = 35 , height = 5)
        self.mtext.grid(row = 0 ,column = 0,columnspan = 2)

        tk.Label(self.frame, text="Tonic:", width=5).grid(row=0, column=30)
        self.tonic = tk.Text(self.frame,width = 5 , height = 3)
        self.tonic .grid(row = 0 ,column = 35,columnspan = 2)
        #print("test1")
        #print(self.tonic)
        tk.Button(self.frame,text = "Fold",command = lambda q=self.tonic:self.fold(q)).grid(row =0,column = 40,columnspan=2)

        tk.Label(self.frame, text="[cents]", width=10).grid(row=4, column=0)
        tk.Label(self.frame, text="#Bin", width=10).grid(row=4, column=5)
        for row in range(240):
            txt=str(5*row)+"-"+str(5*row+4)
            tk.Label(self.frame, text=txt, width=10, borderwidth="1", 
                     relief="solid").grid(row=row+10, column=0)
            tk.Button(self.frame,text = "Bin"+str(row),command = lambda p=row:self.reveal(p)).grid(row =row+10,column =5,columnspan=1)
        
        
	
    def onFrameConfigure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def maxbin(self,low,high):
        while(not self.d[low]):
            low=low+1
        self.max1=max(self.d[low])
        for k in range(int(low),int(high)):
            if(not self.d[k]):
                continue
            else:
                max2=max(self.d[k])    
                self.max1=max(self.max1,max2)



    def fold(self,q):
        #print("test2")
        #print(q)
        tonic =q.get('1.0', tk.END)
        n = round(240*(log(float(tonic)/256)/log(2)),0)%240
        if(n-9<0):
            low=0
        else:
            low=n-9

        if(n+10>240):
            high=240
        else:
            high=n+10
        maxbin(low,high)
        print(self.max1)        
        self.process(self.max1)

    def reveal(self,x):
        self.mtext.delete('1.0', tk.END)
        message=str(self.d[x])
        self.mtext.insert(0.0,message)
        #print (self.d[x])

    def process(self,ftonic):
        thefile = open('test.txt', 'w')
        try:
            filename = sys.argv[1]
        except:
            print("usage: %s <input-audiofile>" % sys.argv[0])        
        text_file = open(filename, "r")
        lines = text_file.read().split('\n')
        text_file.close()
        list1=[]
        list_count=[]
        self.len2=len(lines)
        self.d = defaultdict(list)
        for i in range(len(lines)-1):
            list1.append(lines[i].split())
            n = round(240*(log(float(list1[i][1])/ftonic)/log(2)),0)%240
            self.d[n].append(float(list1[i][1]))
        #Count best 20 bins
            for j in range(len(self.d)):
                list_count.append(len(self.d[j]))
        # plt.plot(list_count)
        # plt.ylabel('Count')
        # plt.xlabel('Bin#')
        # max1=max(list_count)
        # plt.xlim(0, 240)
        # plt.ylim(0, max1)
        for k in range(len(list_count)):
            str1=str(k)+"\t"+str(list_count[k]) +"\n"
            thefile.write(str1)
            #thefile.write("%s\n" % item)
        #plt.show()
        thefile.close()
        lines = open("test.txt", 'r').readlines()
        #output = open("inter.txt", 'w')
        #for line in sorted(lines, key=lambda line: line.split()[1]):
            #output.write(line)
        # os.system("sort -k2n test.txt > inter.txt")   
        os.system("sort -nrk 2,2 test.txt > inter.txt")

        text_file.close()
        #plot 12 swaras
        



    def meanbin(self,low,high):
        sum1=0
        len1=0
        list2=[]
        list3=[]
        self.std1=0
        self.mean1=0  
        for k in range(int(low),int(high)):
        	list2+=self.d[k]
        	sum1+=sum(self.d[k])
        	len1+=len(self.d[k])
        if(len1>0):    
            self.mean1=int(sum1/len1)
        else:
            self.mean1=0
        if(len(list2)>0):        
            self.max1=max(list2)
        else:
            self.max1=0    
        for s in range(len(list2)):
        	x=math.pow((list2[s]-self.mean1),2)
        	list3.append(x)
        if(len(list2)>0):
       	    y=sum(list3)/len(list2)
        else:
            y=0
       	self.std1=int(math.sqrt(y))
        if(len1>0):
            self.prob1=float(len1/self.len2)
        else:
            self.prob1=0

    def PCD(self):
        self.mean=[]
        self.peak=[]
        self.sigma=[]
        self.prob=[]
        for j in range(0,12):
            if(j==0):
                self.meanbin(0,50)
                mean3=self.mean1
                std3=self.std1
                max3=self.max1
                prob3=self.prob1
                
                self.meanbin(1150,1200)
                mean4=self.mean1
                std4=self.std1
                max4=self.max1
                prob4=self.prob1

                self.mean.append((mean3+mean4)//2)
                self.peak.append(max(max3,max4))
                self.prob.append(float((prob3+prob4)/2))
                self.sigma.append(float((std3+std4)/2))
            else:
                low=20*j-10
                high=20*j+10
                self.maxbin(low,high)
                self.peak.append(self.max1)
                self.meanbin(low,high)
                self.mean.append(self.mean1)
                self.sigma.append(self.std1)
                self.prob.append(self.prob1)
        #write output to output file
        f = open('Output', 'w')
        f.write('Bin# \t mean \t sigma \t prob \t peak \n')
        rec=[]  
        line2=[]    
        for k in range(0,12):
            #del line2[:]
            line2=[] 
            print("mean of PCD bin"+str(k)+": "+str(self.mean[k]))
            print("sigma of PCD bin"+str(k)+" "+str(self.sigma[k]))
            print("prob of PCD bin"+str(k)+" "+str(self.prob[k]))
            print("peak of PCD bin"+str(k)+" "+str(self.peak[k]))
            print("=======================================================================")
            line2.append(k)
            line2.append(self.mean[k])
            line2.append(self.sigma[k])
            line2.append(self.prob[k])
            line2.append(self.peak[k])
            rec.append(line2)

            #f.write(str(k)+"\t"+str(self.mean[k])+"\t"+str(self.sigma[k])+"\t"+str(self.prob[k])+"\t"+str(self.peak[k])+"\n")
        #print(rec)
        #f1=open('Output1.csv','ab')
        #np.savetxt(f1, np.array(rec), delimiter=",",fmt='%.5f')
        f.close()

        objects = ('Sa', 're', 'Re', 'ga', 'Ga', 'ma','Ma','Pa','dha','Dha','ni','Ni')
        y_pos = np.arange(len(objects))
        #performance = [10,8,6,4,2,1]
         
        plt.bar(y_pos, self.prob, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Prob')
        plt.ylim(0,1)
        plt.title('Pitch class Distribution')
         
        plt.show()


#main
root=tk.Tk()
Example(root).pack(side="top", fill="both", expand=True)
# 
root.mainloop()
