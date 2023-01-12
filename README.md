# pytorch_chinese_biaffine_ner
使用biaffine的中文命名实体识别。

# 依赖

```python
transformers==4.5.0
```

# 运行



包含训练、验证、测试和预测。

```python
python main.py
```

### 结果

使用的参数：

```python
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
```

```python
【train】Epoch: 20/20 Step: 1200/1200 loss: 0.00156
【eval】precision=0.8722 recall=0.9044 f1_score=0.8880
【best_f1】0.8880258899676375
[eval] precision=0.8854 recall=0.9206 f1_score=0.9027
          precision    recall  f1-score   support

     PRO       0.91      0.82      0.86        35
     ORG       0.87      0.93      0.90       563
    CONT       1.00      1.00      1.00        28
    RACE       1.00      1.00      1.00        14
    NAME       1.00      0.99      1.00       112
     EDU       0.95      0.96      0.96       114
     LOC       0.83      1.00      0.91         6
   TITLE       0.86      0.90      0.88       830

micro-f1       0.89      0.92      0.90      1702

顾建国先生：研究生学历，正高级工程师，现任本公司董事长、马钢(集团)控股有限公司总经理。
torch.Size([1, 150, 150, 9])
{'NAME': [['顾建国', 0, 3]], 'EDU': [['研究生学历', 6, 11]], 'TITLE': [['正高级工程师', 12, 18], ['董事长', 24, 27], ['总经理', 40, 43]], 'ORG': [['本公司', 21, 24], ['马钢(集团)控股有限公司', 28, 40]]}
```

# 补充

Q：怎么训练自己的数据？

A：按照cner里面数据的格式，然后修改main.py里面相关参数。

# 参考

模型代码参考：https://github.com/modelscope/AdaSeq
