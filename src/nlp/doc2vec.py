import pickle
import os
from tqdm.notebook import tqdm
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


def apply(data_path='dataset.txt', save_name='doc2vec_model', size=100, window=3, min_count=2):
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

    # make corpus
    Corpus = []
    idx2doc = {}
    for idx, text in tqdm(enumerate(texts), desc='데이터 로드', leave=False):
        idx2doc[idx] = text
        for t in text:
            Corpus.append(TaggedDocument(words=t, tags=[idx]))

            # training
    print('Doc2vec training...')
    model = Doc2Vec(Corpus, dm=1, vector_size=size, window=window, epochs=10, min_count=min_count, negative_size=5,
                    alpha=0.025, workers=os.cpu_count())

    # save
    model.save(save_name)

    with open(save_name + '_vocab', 'wb') as f:
        pickle.dump(idx2doc, f)

    print('training done!')


def get_similar_doc(doc_index, save_name):
    doc2vec = Doc2Vec.load('doc2vec_model')
    doc2vec.random.seed(2021)

    with open(save_name + '_vocab', 'rb') as f:
        idx2doc = pickle.load(f)

    # vector_text = doc2vec.infer_vector(doc_index.split(), alpha=0.025, min_alpha=0.01, epochs=1000)
    # for e in doc2vec.docvecs.most_similar([vector_text], topn=3):
    #     doc_idx, score = e[0], e[1]
    #     print('doc_id: {}, score: {}, doc: {}'.format(doc_idx, round(score, 2), idx2doc[doc_idx]))

    print('target text: {}:'.format(idx2doc[doc_index]))
    print(' ')
    for e in doc2vec.docvecs.most_similar(doc_index, topn=3):
        doc_idx, score = e[0], e[1]
        print('doc_id: {}, score: {}, doc: {}'.format(doc_idx, round(score, 2), idx2doc[doc_idx]))