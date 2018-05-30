import nltk
import re
#nltk.download('stopwords')

english_stemmer = nltk.stem.SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))

token_pattern = r"(?u)\b\w\w+\b"

test= "Winnie the pooh"

def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed


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

def body_to_sentences(body):
    body=body.replace("?",".")
    body=body.replace("!",".")
    arr=body.split(".")
    while '' in arr: arr.remove('')
    
    for k in range(len(arr)):
        arr[k]=re.sub(r'[^\w\'\s]','',arr[k]).lstrip()
        arr[k]=' '.join(removeStop(arr[k].split(" ")))
    return arr

print(body_to_sentences("whispering windand and don't donrd.the tune will come to you at? here we go, and here."))