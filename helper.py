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

print(stem_tokens(["Winnie","the","pooh","cat","mat","premature","obscure","distortion"],english_stemmer))

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

print(preproc("Winnie the pooh cat mat premature obscure distortion",token_pattern,stopwords,english_stemmer))