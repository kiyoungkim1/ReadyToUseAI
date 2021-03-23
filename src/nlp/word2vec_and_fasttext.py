import os
from gensim.models import Word2Vec, FastText

def apply(data_path='dataset.txt', type='fasttext', save_name=None, size=100, window=3, min_count=2, iter=10):
    assert type in ['word2vec', 'fasttext'], 'type은 word2vec 또는 fasttext만 가능합니다.'

    # get data
    texts = None
    if isinstance(data_path, str):
        # read text file
        with open(data_path, 'r') as f:
            texts = [str(text).replace('\n', '') for text in f.readlines() if len(str(text)) >= 10]
    elif isinstance(data_path, list):
        texts = data_path
    else:
        AssertionError('Invalid data_path type: {}'.format(type(data_path)))

    inputs = [tt.split(' ') for tt in texts]
    print('number of sentences = ', len(inputs))

    # training
    model = None
    if type == 'word2vec':
        print('word2vec training...')
        model = Word2Vec(inputs, size=size, window=window, min_count=min_count, negative=5, workers=os.cpu_count(),
                        iter=iter, sg=1)
    elif type == 'fasttext':
        print('fasttext training...')
        model = FastText(inputs, size=size, window=window, min_count=min_count, negative=5, workers=os.cpu_count(),
                        iter=iter, sg=1)

    model.save(save_name)

    print('training done!')
