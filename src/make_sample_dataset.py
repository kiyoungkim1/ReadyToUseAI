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
