from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def apply(dataset_path='dataset.txt', save_path='count.xlsx'):
    # read dataset
    with open(dataset_path, 'r') as f:
      corpus = [line.replace('\n','').strip() for line in f.readlines()]

    # initialize
    vect = TfidfVectorizer()

    # fit
    vect.fit(corpus)

    # transform
    vec_array = vect.transform(corpus).toarray()

    # save
    columns = [word[0] for word in sorted(vect.vocabulary_.items(), key=lambda x: x[1])]
    df = pd.DataFrame(vec_array, columns=columns)

    total = df.sum().to_frame(name='total').T
    df = pd.concat([total, df])


    df.sort_values(by=['total'], ascending=False, axis=1)
    df.to_excel(save_path)
    print('Result is saved at {}'.format(save_path))

    return df
