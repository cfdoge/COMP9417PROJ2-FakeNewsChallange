import re
import nltk


stopwords = set(nltk.corpus.stopwords.words('english'))


def removeStop(line):
    #print(re.search('\'',"don't"))
    arr=[]
    for k in range (len(line)):
        #print(line[k])
        if re.search('\'',line[k]):
            arr.append(line[k])
        elif line[k] not in stopwords:
            arr.append(line[k])
    return arr

def Handle(words):
    ans =re.split("[ ,.;?!:\"]",words)
    p=0
    while p < len(ans):
        if ans[p]=="":
            ans.pop(p)
        p=p+1
    
    
    return removeStop(ans)

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

print(getGrams("Winnie the pooh? the pooh", "pooh was angry with tiger, and piglet didn't like it",2))