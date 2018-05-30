# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:55:57 2018

@author: Michael-JamesCoetzee
"""

import re, math
import nltk
stopwords = set(nltk.corpus.stopwords.words('english'))

#remove stop words
def removeStop(line,stopwords):
    
    arr=[]
    for k in range (len(line)):
        if re.search('\'',line[k]):
            arr.append(line[k])
        elif line[k] not in stopwords:
            arr.append(line[k])
    return arr

def DensitySearch(words, body, parts,weights):
    body=body.replace("?",".")
    body=body.replace("!",".")
    arr=body.split(".")
    for k in range(len(arr)):
        arr[k] = arr[k].strip()
    while '' in arr: arr.remove('')
    
    # size of each part
    gap=math.floor(len(arr)/parts)
    
    
    h=0
    body=[]
    
    #create body parts with punctuation and stopwords removed
    for m in range(parts-1):
        temp=[]
        for k in range(h,h+gap):
            arr[k] = re.sub(r'[^\w\'\s]','',arr[k])
            temp.append(removeStop(arr[k].strip().split(),stopwords))
        h=h+gap
        body.append(temp)
    temp2=[]
    
    #last part maybe a few lines longer if parts doesnt divide len(arr)
    for b in range(h,len(arr)):
        arr[b] = re.sub(r'[^\w\'\s]','',arr[b])
        temp2.append(removeStop(arr[b].strip().split(),stopwords))
    body.append(temp2)
    
    sect=-1
    max=0
    found=False
    
    #find body part with maximum weighted sum
    for p in range(len(body)):
        count=0# set weighted sum to 0 for each new body part
        for k in range(len(body[p])):
            for j in range(len(body[p][k])):
                for h in range(len(words)):
                    if body[p][k][j]==words[h]:
                        count=count+weights[h]
        if count>max:
            max=count
            sect=p
            found=True
    
    ans=[]
    if found:
        for p in range(len(body[sect])):
            ans.append(' '.join(body[sect][p]))
        return ans
    else:
        return False


body="There's a lady who's sure all that glitters is gold. And she's buying a stairway to heaven. When she gets there she knows, if the stores are all close With a word she can get what she came"


print(DensitySearch(["glitters","gold"],body,3,[6,4]))