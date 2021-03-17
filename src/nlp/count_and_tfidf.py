from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np


def apply(dataset_path='dataset.txt', type='count', save_path=None, sort=True):
    assert type in ['count', 'tfidf'], 'type은 count 또는 tfidf만 가능합니다.'

    # read dataset
    with open(dataset_path, 'r') as f:
        corpus = [line.replace('\n', '').strip() for line in f.readlines()]

    # initialize
    vect = CountVectorizer() if type=='count' else TfidfVectorizer()

    # fit
    vect.fit(corpus)

    # transform
    vec_array = vect.transform(corpus).toarray()

    # save
    columns = [word[0] for word in sorted(vect.vocabulary_.items(), key=lambda x: x[1])]
    df = pd.DataFrame(vec_array, index=corpus, columns=columns)

    total = df.sum().to_frame(name='total').T
    df = pd.concat([total, df])

    if sort:
        df.sort_values(by=['total'], ascending=False, axis=1, inplace=True)

    if save_path:
        df.to_excel(save_path)
        print('Result is saved at {}'.format(save_path))

    return df, vect


def cosine_similarity(a, b):
    sim = np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))
    return sim[0]


def find_similar_sentence(text, dataset_path='dataset.txt', type='count', save_path=None):
    df, vect = apply(dataset_path=dataset_path, type=type, save_path=None, sort=False)
    if 'total' in df.index:
        df.drop(index='total', inplace=True)

    # get cosine similarity
    text_tokenized = vect.transform([text]).toarray()
    df['similarity'] = df.index.map(lambda x: cosine_similarity(text_tokenized, df.loc[x, :].values))

    # similarity column as first column
    col = list(df.columns)
    value = col.pop()
    col.insert(0, value)

    df = df[col]

    # sort by similarity
    df.sort_values(by=['similarity'], ascending=False, axis=0, inplace=True)

    if save_path:
        df.to_excel(save_path)
        print('Result is saved at {}'.format(save_path))

    return df