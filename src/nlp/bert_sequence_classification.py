"""
This is somewhat old version code. Work well but will be revisited.
"""

import os
import numpy as np
import pandas as pd
from datetime import timedelta
import time
from random import random
import random
from collections import Counter

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizerFast, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))  # 반올림
    return str(timedelta(seconds=elapsed_rounded))  # hh:mm:ss으로 형태 변경

class Classification:
    def __init__(self, model_name='kykim/bert-kor-base', model_dir='model', min_sentence_length=1,
                 MAX_LEN=256, batch_size=64, use_bert_tokenizer=False):

        # Init variable.
        self.model_name = model_name
        self.MAX_LEN = MAX_LEN  # 입력 토큰의 최대 시퀀스 길이
        self.batch_size = batch_size  # 배치 사이즈
        self.min_sentence_length = min_sentence_length

        # path
        self.model_dir = model_dir
        self.save_path = os.path.join(model_dir, 'saved')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # 디바이스 설정
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # print('There are %d GPU(s) available.' % torch.cuda.device_count())
            # print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            print('No GPU available, using the CPU instead.')

        # MODEL
        self.tokenizer_class = AutoTokenizer
        self.model_class = AutoModelForSequenceClassification

        if use_bert_tokenizer:
            self.tokenizer_class = BertTokenizerFast

    def dataset(self, data_path):
        data = pd.read_excel(data_path)

        self.sentences, self.labels = [], []
        for idx in data.index:
            sentence = data.loc[idx, 'content']
            label = data.loc[idx, 'label']

            if len([i for i in str(sentence) if i != ' ']) < self.min_sentence_length:
                continue
            if not np.isnan(label):
                self.labels.append(int(label))
                self.sentences.append(str(sentence))
            else:
                print('label error:', sentence, label)

        self. num_labels = len(list(set(self.labels)))
        print('{} labels, {} dataset'.format(self.num_labels, len(self.labels)))
        print('label counts:: {}'.format(Counter(self.labels)))

    def load_model(self, mode=None, saved_model_path=None):
        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_name)

        if mode == 'train':
            self.model = self.model_class.from_pretrained(self.model_name, num_labels=self.num_labels)
        elif mode == 'inference':
            self.model = self.model_class.from_pretrained(saved_model_path)

        if self.device == torch.device("cuda"):
            self.model.cuda()

    def tokenizing(self, mode='train', dataset_split=0.1):
        if mode == 'train':
            input_ids = [self.tokenizer.encode(sentence, padding='max_length',
                                               max_length=self.MAX_LEN,
                                               truncation=True,
                                               ) for sentence in self.sentences]

            # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
            attention_masks = []
            for seq in input_ids:
                seq_mask = [float(i > 0) for i in seq]
                attention_masks.append(seq_mask)

            # 훈련셋과 검증셋으로 분리
            train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, self.labels,
                                                                                                random_state=2021,
                                                                                                test_size=dataset_split)

            train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2021,
                                                                   test_size=dataset_split)

            # 데이터를 파이토치의 텐서로 변환
            train_inputs = torch.tensor(train_inputs)
            train_labels = torch.tensor(train_labels)
            train_masks = torch.tensor(train_masks)
            validation_inputs = torch.tensor(validation_inputs)
            validation_labels = torch.tensor(validation_labels)
            validation_masks = torch.tensor(validation_masks)

            train_data = TensorDataset(train_inputs, train_masks, train_labels)
            train_sampler = RandomSampler(train_data)
            self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
            validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
            validation_sampler = SequentialSampler(validation_data)
            self.validation_dataloader = DataLoader(validation_data, sampler=validation_sampler,
                                                    batch_size=self.batch_size)
            print('{}-dataset is prepared'.format(mode))

        elif mode == 'inference':
            input_ids = [self.tokenizer.encode(sentence, padding='max_length',
                                               max_length=self.MAX_LEN,
                                               truncation=True,
                                               ) for sentence in self.sentences]

            # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
            attention_masks = []
            for seq in input_ids:
                seq_mask = [float(i > 0) for i in seq]
                attention_masks.append(seq_mask)

            inputs = torch.tensor(input_ids).to(self.device)
            masks = torch.tensor(attention_masks).to(self.device)

            return inputs, masks

    def inference(self, sentences):
        self.sentences = sentences

        model = self.model
        model.eval()

        # tokenizer
        b_input_ids, b_input_mask = self.tokenizing(mode='inference')

        try:
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        except Exception as E:
            print(E)
            exit()
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()

        return [np.argmax(logit) for logit in logits]


    def train(self, epochs=1, log_dir='log', dataset_split=0.1):
        # tokenizer
        self.tokenizing(mode='train', dataset_split=dataset_split)

        writer = SummaryWriter(log_dir)

        model = self.model
        optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)

        # 총 훈련 스텝 : 배치반복 횟수 * 에폭
        total_steps = len(self.train_dataloader) * epochs

        # 학습률을 조금씩 감소시키는 스케줄러 생성
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # 재현을 위해 랜덤시드 고정
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # 그래디언트 초기화
        model.zero_grad()

        # 에폭만큼 반복
        for epoch_i in range(0, epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
            t0 = time.time()
            total_loss = 0
            train_accuracy, nb_train_steps = 0, 0

            model.train()

            # 데이터로더에서 배치만큼 반복하여 가져옴
            for step, batch in enumerate(self.train_dataloader):
                # 경과 정보 표시
                if step % 100 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader),
                                                                                elapsed))

                batch = tuple(t.to(self.device) for t in batch)  # 배치를 GPU에 넣음
                b_input_ids, b_input_mask, b_labels = batch  # 배치에서 데이터 추출
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                                labels=b_labels)  # Forward 수행
                loss = outputs[0]  # 로스 구함
                total_loss += loss.item()  # 총 로스 계산

                loss.backward()  # Backward 수행으로 그래디언트 계산
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 그래디언트 클리핑
                optimizer.step()  # 그래디언트를 통해 가중치 파라미터 업데이트

                scheduler.step()  # 스케줄러로 학습률 감소
                model.zero_grad()  # 그래디언트 초기화

                ##accuracy
                logits = outputs[1]
                # CPU로 데이터 이동
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # 출력 로짓과 라벨을 비교하여 정확도 계산
                train_accuracy += flat_accuracy(logits, label_ids)
                nb_train_steps += 1

            # 평균 로스 계산
            avg_train_loss = total_loss / len(self.train_dataloader)

            print("")
            print("  Train loss: {0:.2f}, Train Accuracy: {1:.2f}".format(avg_train_loss,
                                                                          train_accuracy / nb_train_steps))
            print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            t0 = time.time()
            model.eval()

            # 변수 초기화
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps = 0
            labels_accuracy, preds_accuracy = [], []

            # 데이터로더에서 배치만큼 반복하여 가져옴
            for batch in self.validation_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

                logits = outputs[0]
                # CPU로 데이터 이동
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # 출력 로짓과 라벨을 비교하여 정확도 계산
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

                labels_accuracy.append(label_ids.flatten())
                preds_accuracy.append(np.argmax(logits, axis=1).flatten())

            print("  Validation Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print("  Validation took: {:}".format(format_time(time.time() - t0)))

            # precision and recall
            labels_accuracy = [y for x in labels_accuracy for y in x]  # list flatten
            preds_accuracy = [y for x in preds_accuracy for y in x]
            print(classification_report(labels_accuracy, preds_accuracy))

            writer.add_scalar('Avg_loss(training)', avg_train_loss, epoch_i + 1)
            writer.add_scalars('Accuracy', {'Train': train_accuracy / nb_train_steps,
                                            'Val': eval_accuracy / nb_eval_steps}, epoch_i + 1)

            if (epoch_i + 1) % 3 == 0 and (epoch_i + 1) != epochs:  ##마지막 iteration은 아래에서 수행.
                save_path = os.path.join(self.save_path, str(epoch_i + 1))
                if not os.path.exists(save_path): os.makedirs(save_path)

                model.save_pretrained(save_path)

        print("")
        print("Training complete!")
        writer.close()

        save_path = os.path.join(self.save_path, str(epochs))
        if not os.path.exists(save_path): os.makedirs(save_path)

        model.save_pretrained(save_path)
