# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:35:25 2018

@author: Michael-JamesCoetzee
"""

import re, math



english_stemmer = nltk.stem.SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))

token_pattern = r"(?u)\b\w\w+\b"

def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

def preproc(line,
                    token_pattern=token_pattern,
                    exclude_stopword=True,
                    stem=True):
    token_pattern = re.compile(token_pattern)
    tokens = [x.lower() for x in token_pattern.findall(line)]
    tokens_stemmed = tokens
    if stem:
        tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]

    return tokens_stemmed

def removeStop(line,stopwords):
    #print(re.search('\'',"don't"))
    arr=[]
    for k in range (len(line)):
        if re.search('\'',line[k]):
            arr.append(line[k])
        elif line[k] not in stopwords:
            arr.append(line[k])
    return arr

def DensitySearch(word, file, parts):
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
            #print("before",arr[k])
            arr[k] = re.sub(r'[^\w\'\s]','',arr[k])
            #print("after",arr[k])
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
                if body[p][k][j]==word:
                    count=count+1
        if count>max:
            max=count
            sect=p
    #print(body[sect])
    ans=""
    for p in range(len(body[sect])):
        ans+=' '.join(body[sect][p])
    return ans
            
def searchDense(word, file, parts):
    f = open(file,'r')
    arr= f.readlines()
    gap=math.floor(len(arr)/parts)
    body=[]
    for k in range(len(arr)):
        arr[k] = re.sub(r'[^\w\s]','',arr[k])
        body.append(arr[k].strip().split(" "))

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
            
print(DensitySearch("lady","densTest.txt",3))