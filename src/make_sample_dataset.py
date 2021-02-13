import requests
from io import StringIO
import pandas as pd

def save_dataset(corpus, dataset_path):
    with open(dataset_path, 'w') as f:
      for line in corpus:
          f.write(str(line)+'\n')

def simple(dataset_path='dataset.txt'):
    corpus = [
        '학교에 가서 수업을 들었다. 학교에 간건 오랜만이다.',
        '학교에 가서 친구 얘기를 들었다.',
        '내일 가서 뭐 먹지?'
    ]

    save_dataset(corpus, dataset_path)
    return corpus


def nsmc(mode='test', text_only=False):
  """
    mode: ['train' or 'test'] Dataset type
    text_only: [bool]
  """
  res = requests.get('https://raw.githubusercontent.com/e9t/nsmc/master/ratings_{}.txt'.format(mode))
  df = pd.read_csv(StringIO(res.text), sep='\t')

  if text_only:
    with open('dataset.txt', 'w') as f:
      for text in list(df['document']):
        f.write(str(text) + '\n')
  else:
    df.to_excel('dataset.xlsx')
