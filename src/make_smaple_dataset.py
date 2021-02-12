# sample dataset

corpus = [
    '학교에 가서 수업을 들었다. 학교에 간건 오랜만이다.',
    '학교에 가서 친구 얘기를 들었다.',
    '내일 가서 뭐 먹지?'
]

with open('dataset.txt', 'w') as f:
  for line in corpus:
      f.write(str(line)+'\n')