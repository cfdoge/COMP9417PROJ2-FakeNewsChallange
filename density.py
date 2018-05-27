# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:35:25 2018

@author: Michael-JamesCoetzee
"""

import re, math

def searchDense(word, file, parts):
    f = open(file,'r')
    PARTS=3
    arr= f.readlines()
    gap=math.floor(len(arr)/parts)
    body=[]
    for k in range(len(arr)):
        arr[k] = re.sub(r'[^\w\s]','',arr[k])
        body.append(arr[k].strip().lower().split(" "))

    ans=[]
    #print(body)
    dens=[0]*(parts+1)
    #print(dens)
    for p in range(len(body)):
        for k in range(len(body[p])):
            if body[p][k]==word:
                #print(p,k)
                dens[math.floor(p/gap)]=1+dens[math.floor(p/gap)]
                ans.append([math.floor(p/gap),p%gap])
                #print(int(p/gap),k)
    #print(dens)
    #print(ans)
    
    pos=0
    max=0
    answer=(0,0)
    for k in range(len(dens)):
        if dens[k]>max:
            max=dens[k]
            answer=(pos,dens[k])
        pos=pos+1
    return answer
            
print(searchDense("lady","densTest.txt",3))