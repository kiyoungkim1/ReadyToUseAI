import os
from gensim.models import Word2Vec

def word2vec(texts, save_name='word2vec', size=100, window=3, min_count=2, iter=10):
    inputs = [tt.split(' ') for tt in texts]
    print('number of sentences = ', len(inputs))

    print('word2vec training...')
    model = Word2Vec(inputs, size=size, window=window, min_count=min_count, negative=5, workers=os.cpu_count(),
                     iter=iter, sg=1)
    model.save(save_name)

    print('training done!')