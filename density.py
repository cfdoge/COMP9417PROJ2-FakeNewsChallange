# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:35:25 2018

@author: Michael-JamesCoetzee
"""



def searchDense(word, file, parts):
    f = open(file,'r')
    PARTS=3
    arr= f.readlines()
    gap=int(len(arr)/parts)
    body=[]
    for k in range(len(arr)):
        body.append(arr[k].strip().lower().split(" "))

    ans=[]
    for p in range(len(body)):
        for k in range(len(body[p])):
            if body[p][k]==word:
                #print(p,k)
                ans.append([int(p/gap),p-int(p/gap)*gap])
                #print(int(p/gap),k)
    return ans
            
            
print(searchDense("lady","densTest.txt",3))