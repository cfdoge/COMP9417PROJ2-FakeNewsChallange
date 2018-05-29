# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:55:57 2018

@author: Michael-JamesCoetzee
"""

import re, math

stopwords = set(nltk.corpus.stopwords.words('english'))

def removeStop(line,stopwords):
    #print(re.search('\'',"don't"))
    arr=[]
    for k in range (len(line)):
        if re.search('\'',line[k]):
            arr.append(line[k])
        elif line[k] not in stopwords:
            arr.append(line[k])
    return arr

def DensitySearch(words, file, parts,weights):
    f=open(file,'r')
    arr=f.readlines()
    #print(arr)
    for k in range(len(arr)):
        arr[k] = arr[k].strip()
    while '' in arr: arr.remove('')
    #print(arr)
    gap=math.floor(len(arr)/parts)
    #print(len(arr))
    #print(gap*parts)
    h=0
    body=[]
    for m in range(parts-1):
        temp=[]
        for k in range(h,h+gap):
            arr[k] = re.sub(r'[^\w\'\s]','',arr[k])
            temp.append(removeStop(arr[k].strip().split(),stopwords))
        h=h+gap
        body.append(temp)
    temp2=[]
    for b in range(h,len(arr)):
        arr[b] = re.sub(r'[^\w\'\s]','',arr[b])
        temp2.append(removeStop(arr[b].strip().split(),stopwords))
    body.append(temp2)
    #print(body)
    sect=-1
    max=0
    for p in range(len(body)):
        count=0
        for k in range(len(body[p])):
            for j in range(len(body[p][k])):
                for h in range(len(words)):
                    if body[p][k][j]==words[h]:
                        count=count+weights[h]
        if count>max:
            max=count
            sect=p
    #print(body[sect])
    ans=[]
    for p in range(len(body[sect])):
        ans.append(' '.join(body[sect][p]))
    return ans

print(DensitySearch(["lady","head"],"densTest.txt",3,[6,4]))