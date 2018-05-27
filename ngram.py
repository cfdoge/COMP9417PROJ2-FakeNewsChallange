import re
import nltk

def Handle(words):
    ans =re.split("[ ,.;?!:\"]",words)
    p=0
    while p < len(ans):
        if ans[p]=="":
            ans.pop(p)
        p=p+1
    return ans

def Unigram(words):
    return words

def Bigram(words):
    ans=[]
    for k in range(len(words)-1):
        ans.append(words[k]+"_"+words[k+1])
    return ans

def Trigram(words):
    ans=[]
    for k in range(len(words)-2):
        ans.append(words[k]+"_"+words[k+1]+"_"+words[k+2])
    return ans

def getGrams(title, main, type):
    if type==1:
        return(Unigram(Handle(title)),Unigram(Handle(main)), len(set(Unigram(Handle(title)))),len(set(Unigram(Handle(main)))))
    if type==2:
        return (Bigram(Handle(title)), Bigram(Handle(main)), len(set(Bigram(Handle(title)))),len(set(Bigram(Handle(main)))))
    if type==3:
        return (Trigram(Handle(title)), Trigram(Handle(main)), len(set(Trigram(Handle(title)))),len(set(Trigram(Handle(main)))))
    return([],[],0)

print(getGrams("Winnie the pooh the pooh", "pooh was angry with tiger, and piglet didn't like it",1))