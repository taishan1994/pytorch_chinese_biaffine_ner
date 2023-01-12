import os
import json
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from data_loader import CnerDataset, Collate
from model import BiaffineModel
from utils import *


def pair_decode(pair_pred, text, id2label):
  res = defaultdict(list)
  for i in range(len(pair_pred)):
    for j in range(i, len(pair_pred[0])):
      if pair_pred[i][j] > 0:
        res[id2label[pair_pred[i][j]]].append(["".join(text[i:j+1]), i-1, j])
        break
  return dict(res)


class NerPipeline:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def save_model(self):
        torch.save(self.model.state_dict(), self.args.save_dir)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.save_dir, map_location="cpu"))

    def build_optimizer_and_scheduler(self, t_total):
        module = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        # 差分学习率
        no_decay = ["bias", "LayerNorm.weight"]
        model_param = list(module.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []

        for name, para in model_param:
            space = name.split('.')
            # print(name)
            if "bert" in space[0]:
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # bert other module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.lr},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.lr},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.other_lr},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(self.args.warmup_proportion * t_total), num_training_steps=t_total
        )

        return optimizer, scheduler

    def eval_forward(self, data_loader):
        span_logits = None
        span_labels = None
        self.model.eval()
        for eval_step, batch_data in enumerate(data_loader):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(self.args.device)
            labels = batch_data["labels"]
            output = model(batch_data,
                    labels)
            labels = labels.detach().cpu().numpy()
            span_logit, loss = output
            span_logit = span_logit.detach().cpu().numpy()
            span_logit = np.argmax(span_logit, -1)
            if span_logits is None:
              span_logits = span_logit
              span_labels = labels
            else:
              span_logits = np.append(span_logits, span_logit, axis=0)
              span_labels = np.append(span_labels, labels, axis=0)

        return span_logits, span_labels

    def get_metric(self, span_logits, span_labels, callback):
        batch_size = len(callback)
        total_count = [0 for _ in range(len(self.args.id2tag))]
        role_metric = np.zeros([len(self.args.id2tag), 3])
        for span_logit, label, tokens in zip(span_logits, span_labels, callback):
          pred_entities = pair_decode(span_logit, tokens, self.args.id2tag)
          gt_entities = pair_decode(label, tokens, self.args.id2tag)
          # print("========================")
          # print(pred_entities)
          # print(gt_entities)
          # print("========================")
          for idx, _type in enumerate(list(self.args.tag2id.keys())):
              if _type not in pred_entities:
                  pred_entities[_type] = []
              if _type not in gt_entities:
                  gt_entities[_type] = []
              total_count[idx] += len(gt_entities[_type])
              role_metric[idx] += calculate_metric(pred_entities[_type], gt_entities[_type])
                                                  
        return role_metric, total_count

    def train(self, dev=True):
        train_dataset, train_callback = CnerDataset(file_path=self.args.train_path,
                                tokenizer=self.args.tokenizer,
                                max_len=self.args.max_seq_len,)
        collate = Collate(max_len=self.args.max_seq_len, tag2id=self.args.tag2id)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2,
                                  collate_fn=collate.collate_fn)
        dev_loader = None
        dev_callback = None
        if dev:
            dev_dataset, dev_callback = CnerDataset(file_path=self.args.dev_path,
                                tokenizer=self.args.tokenizer,
                                max_len=self.args.max_seq_len,)
        
            dev_loader = DataLoader(dataset=dev_dataset,
                                    batch_size=self.args.eval_batch_size,
                                    shuffle=False,
                                    num_workers=2,
                                    collate_fn=collate.collate_fn)

        t_total = len(train_loader) * self.args.train_epoch
        optimizer, scheduler = self.build_optimizer_and_scheduler(t_total)

        global_step = 0
        self.model.zero_grad()
        self.model.to(self.args.device)
        eval_step = self.args.eval_step
        best_f1 = 0.
        for epoch in range(1, self.args.train_epoch + 1):
            for step, batch_data in enumerate(train_loader):
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.args.device)
                labels = batch_data["labels"]
                output = self.model(batch_data,
                           labels)
                _, loss = output
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                global_step += 1
                print('【train】Epoch: %d/%d Step: %d/%d loss: %.5f' % (
                    epoch, self.args.train_epoch, global_step, t_total, loss.item()))
                if dev and global_step % eval_step == 0:
                    span_logits, span_labels = self.eval_forward(dev_loader)
                    role_metric, _ = self.get_metric(span_logits, span_labels, dev_callback)
                    mirco_metrics = np.sum(role_metric, axis=0)
                    mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
                    print('【eval】precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(mirco_metrics[0],
                                                      mirco_metrics[1],
                                                      mirco_metrics[2]))
                    if mirco_metrics[2] > best_f1:
                        best_f1 = mirco_metrics[2]
                        print("【best_f1】{}".format(mirco_metrics[2]))
                        self.save_model()

    def test(self):
        test_dataset, test_callback = CnerDataset(file_path=self.args.test_path,
                                                 tokenizer=self.args.tokenizer,
                                                 max_len=self.args.max_seq_len,)
        collate = Collate(max_len=self.args.max_seq_len, tag2id=self.args.tag2id)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 collate_fn=collate.collate_fn)
        self.load_model()
        self.model.to(self.args.device)
        with torch.no_grad():
            span_logits, span_labels = self.eval_forward(test_loader)
            role_metric, total_count = self.get_metric(span_logits, span_labels, test_callback)
            mirco_metrics = np.sum(role_metric, axis=0)
            mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
            print(
                '[eval] precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]))
            label_list = list(self.args.tag2id.keys())
            id2label = {i:label for i,label in enumerate(label_list)}
            print(classification_report(role_metric, label_list, id2label, total_count))

    def predict(self, text):
        self.load_model()
        self.model.eval()
        self.model.to(self.args.device)
        with torch.no_grad():
            tokens = [i for i in text]

            encode_dict = self.args.tokenizer.encode_plus(text=tokens,
                                    max_length=self.args.max_seq_len,
                                    padding="max_length",
                                    truncating="only_first",
                                    return_token_type_ids=True,
                                    return_attention_mask=True)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0).to(self.args.device)
            attention_mask = torch.from_numpy(np.array(encode_dict['attention_mask'])).unsqueeze(0).to(
                self.args.device)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0).to(self.args.device)
            inputs = {
              "input_ids": token_ids,
              "attention_mask": attention_mask,
              "token_type_ids": token_type_ids
            }
            output = self.model(inputs)
            span_logits, _ = output
            print(span_logits.shape)
            span_logits = np.argmax(span_logits.detach().cpu().numpy(), -1)
            res = pair_decode(span_logits[0], tokens, self.args.id2tag)
            return res


if __name__ == '__main__':
  class Args:
    data_name = "cner"
    data_dir = 'data/{}/'.format(data_name)
    train_path = os.path.join(data_dir, "train.json")
    dev_path = os.path.join(data_dir, "dev.json")
    test_path = os.path.join(data_dir, "test.json")
    bert_dir = "model_hub/chinese-bert-wwm-ext"
    save_dir = "checkpoints/{}/model.pt".format(data_name)
    ffnn_size = 256
    max_seq_len = 150
    train_epoch = 20
    train_batch_size = 64
    eval_batch_size= 32
    eval_step = 100
    lr = 3e-5
    other_lr = 2e-3
    adam_epsilon = 1e-8
    warmup_proportion = 0.1
    max_grad_norm = 5
    weight_decay = 0.01
    num_cls = 9
    bias = True

  args = Args()
  with open(os.path.join(args.data_dir, 'labels.json')) as fp:
    labels = json.load(fp)
  id2tag = {}
  tag2id = {}
  for i,label in enumerate(labels):
    id2tag[i+1] = label
    tag2id[label] = i+1
  args.tag2id = tag2id
  args.id2tag = id2tag

  tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
  args.tokenizer = tokenizer
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  args.device = device

  model = BiaffineModel(args)
  model.to(device)
  ner_pipeline = NerPipeline(model, args)

  ner_pipeline.train()
  ner_pipeline.test()

  raw_text = "顾建国先生：研究生学历，正高级工程师，现任本公司董事长、马钢(集团)控股有限公司总经理。"
  print(raw_text)
  print(ner_pipeline.predict(raw_text))