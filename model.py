from typing import Dict, Optional
import os
import json

import torch
import torch.nn as nn

from transformers import BertModel, BertConfig

class Biaffine(torch.nn.Module):
  """Biaffine Attention"""

  def __init__(self, in_features: int, out_features: int, bias=(True, True)):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.bias = bias

    self.linear_input_size = in_features + bias[0]
    self.linear_output_size = out_features * (in_features + bias[1])

    self.linear = torch.nn.Linear(
        in_features=self.linear_input_size, out_features=self.linear_output_size, bias=False
    )

  def forward(self, input1, input2):
    # input1: [64, 150, 256]
    batch_size, len1, dim1 = input1.size()
    # input2: [64, 150, 256]
    batch_size, len2, dim2 = input2.size()

    if self.bias[0]:
        ones = input1.data.new_ones(batch_size, len1, 1)
        # input1: [64, 150, 257]
        input1 = torch.cat((input1, ones), dim=-1)
    if self.bias[1]:
        ones = input2.data.new_ones(batch_size, len2, 1)
        # input2: [64, 150, 257]
        input2 = torch.cat((input2, ones), dim=-1)

    # [64, 150, 257] -> [64, 150, 9 * 257]
    affine = self.linear(input1)

    # [64, 150 * 9, 257]
    affine = affine.reshape(batch_size, len1 * self.out_features, -1)

    # [64, 150 * 9, 257] * [64, 257, 150] = [64, 150*9, 150]
    # [64, 150*9, 150] -> [64, 150, 150*9]
    biaffine = torch.bmm(affine, input2.transpose(1, 2)).transpose(1, 2).contiguous()

    # [64, 150, 150*9] -> [[64, 150, 150, 9]]
    biaffine = biaffine.reshape((batch_size, len2, len1, -1)).squeeze(-1)

    return biaffine


class BiaffineModel(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.bert_config = BertConfig.from_pretrained(args.bert_dir)
    self.bert = BertModel.from_pretrained(args.bert_dir)

    self._act = nn.ELU()
    self.num_cls = args.num_cls
    input_size = self.bert_config.hidden_size
    ffnn_size = args.ffnn_size
    self.mlp_start = nn.Linear(input_size, ffnn_size)
    self.mlp_end = nn.Linear(input_size, ffnn_size)
    # num_cls: 8+1=9
    self.span_biaff = Biaffine(ffnn_size, args.num_cls, bias=(args.bias, args.bias))

  
  @staticmethod
  def build_dummpy_inputs():
    inputs = {}
    inputs['input_ids'] = torch.LongTensor(
        torch.randint(low=1, high=10, size=(64, 150)))
    inputs['attention_mask'] = torch.ones(size=(64, 150)).long()
    inputs['token_type_ids'] = torch.zeros(size=(64, 150)).long()
    inputs['labels'] = torch.zeros(size=(64, 150, 150)).long()
    return inputs

  
  def forward(self, inputs, labels=None):
    """
      inputs:
        input_ids,
        attention_mask,
        tokentype_ids,
    """
    bert_output = self.bert(input_ids=inputs["input_ids"], 
                 attention_mask=inputs["attention_mask"],
                 token_type_ids=inputs["token_type_ids"],)
    bert_output = bert_output[0]
    start_feat = self._act(self.mlp_start(bert_output))
    end_feat = self._act(self.mlp_end(bert_output))
    span_scores = self.span_biaff(start_feat, end_feat)
    if labels is None:
      return span_scores, None
    else:
      loss = nn.functional.cross_entropy(
          span_scores.reshape(-1, self.num_cls),
          labels.reshape(-1),
      )
      return span_scores, loss


if __name__ == "__main__":
  input1 = torch.rand((64, 150, 256))
  input2 = torch.rand((64, 150, 256))

  output = Biaffine(256, 9)(input1, input2)
  print(output.shape)

  class Args:
    bert_dir = "model_hub/chinese-bert-wwm-ext"
    ffnn_size = 256
    num_cls = 9
    bias = True
  
  args = Args()

  inputs = BiaffineModel.build_dummpy_inputs()
  model = BiaffineModel(args)
  model(inputs, labels=inputs["labels"])

  

