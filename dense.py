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

def DensitySearch(words, body, parts,weights):
    #f=open(file,'r')
    body=body.replace("?",".")
    body=body.replace("!",".")
    arr=body.split(".")
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
    found=False
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
            found=True
    #print(body[sect])
    ans=[]
    if found:
        for p in range(len(body[sect])):
            ans.append(' '.join(body[sect][p]))
        return ans
    else:
        return False


body="There's a lady who's sure all that glitters is gold. And she's buying a stairway to heaven. When she gets there she knows, if the stores are all close With a word she can get what she came"


print(DensitySearch(["glitters","gold"],body,3,[6,4]))