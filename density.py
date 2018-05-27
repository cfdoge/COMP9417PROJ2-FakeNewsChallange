# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:35:25 2018

@author: Michael-JamesCoetzee
"""

f = open('densTest.txt','r')
PARTS=3
arr= f.readlines()
gap=int(len(arr)/3)
ans=[]
for k in range(len(arr)):
    ans.append(arr[k].strip().lower().split(" "))

print(ans)


def searchDense(body, word):
    ans=[]
    for p in range(len(body)):
        for k in range(len(body[p])):
            if body[p][k]==word:
                print(p,k)
                ans.append([int(p/gap),p-int(p/gap)*gap])
                print("Andy:",int(p/gap),k)
    return ans
            
            
print(searchDense(ans,"lady"))