import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
# [MODIFIED] 替换 T5Tokenizer 为 RobertaTokenizer 兼容 CodeT5 tokenizer
from transformers import RobertaTokenizer  # [MODIFIED]

class CodeT5Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5Model, self).__init__()
        self.encoder = encoder  # 预训练的 CodeT5
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.query = 0  # 记录查询次数
        self.classifier = nn.Linear(config.d_model, 2)  # 二分类头（CodeT5的hidden_size是d_model）
        self.loss_func = nn.CrossEntropyLoss()

    # def get_t5_vec(self, source_ids):
    #     """
    #     提取 CodeT5 decoder 最后一层 <eos> token 的向量
    #     """
    #     eos_token_id = self.tokenizer.eos_token_id  # [MODIFIED]
    #     attention_mask = source_ids.ne(self.tokenizer.pad_token_id)

    #     # [MODIFIED] 添加 decoder_input_ids（避免 T5 报错）
    #     decoder_input_ids = torch.full(
    #         (source_ids.size(0), 1),
    #         self.tokenizer.pad_token_id,
    #         dtype=torch.long,
    #         device=source_ids.device
    #     )  # [MODIFIED]

    #     outputs = self.encoder(
    #         input_ids=source_ids,
    #         attention_mask=attention_mask,
    #         decoder_input_ids=decoder_input_ids,  # [MODIFIED]
    #         output_hidden_states=True,
    #         return_dict=True
    #     )

    #     hidden_states = outputs.decoder_hidden_states[-1]
    #     eos_mask = (source_ids == eos_token_id)
    #     fallback = torch.zeros_like(eos_mask).scatter_(1, (attention_mask.sum(dim=1) - 1).unsqueeze(1), 1).bool()
    #     final_mask = torch.where(eos_mask.any(dim=1, keepdim=True), eos_mask, fallback)
    #     vec = hidden_states[final_mask].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
    #     return vec
    
    def get_t5_vec(self, source_ids):
        """
        提取 CodeT5 encoder 输出的代码表示向量
        """
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        encoder_outputs = self.encoder.encoder(
            input_ids=source_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        # 平均池化所有非PAD token
        vec = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return vec


    def forward(self, input_ids=None, labels=None, inputs_embeds=None, attention_mask=None):
        """
        前向传播：支持 input_ids 或 inputs_embeds
        """
        if input_ids is not None:

            # 新增
            total_size = input_ids.numel()
            block_size = self.args.block_size
            # 确保 total_size 是 block_size 的整数倍
            new_size = (total_size // block_size) * block_size
            if total_size != new_size:
                print(f"[WARNING] Trimming input_ids from {total_size} to {new_size} for block_size={block_size}")
                input_ids = input_ids.view(-1)[:new_size]

            input_ids = input_ids.view(-1, block_size)
            # 新增

            input_ids = input_ids.view(-1, self.args.block_size)
            vec = self.get_t5_vec(input_ids)
        elif inputs_embeds is not None:
            # [MODIFIED] 添加 decoder_input_ids 避免报错
            decoder_input_ids = torch.full(
                (inputs_embeds.size(0), 1),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=inputs_embeds.device
            )  # [MODIFIED]

            outputs = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,  # [MODIFIED]
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.decoder_hidden_states[-1]
            vec = hidden_states[:, -1, :]  # 默认取最后一个 decoder token
        else:
            raise ValueError("forward() requires either input_ids or inputs_embeds")

        logits = self.classifier(vec)
        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            loss = self.loss_func(logits, labels)
            return loss, prob
        else:
            return prob, logits

    def get_results(self, dataset, batch_size):
        """
        对数据集进行评估，返回概率和标签
        """
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4, pin_memory=False)

        self.eval()
        probs_all = []
        preds_all = []

        for batch in eval_dataloader:
            input_ids = batch[0].to(self.args.device)
            with torch.no_grad():
                prob, _ = self.forward(input_ids=input_ids)
                probs_all.extend(prob.cpu().numpy())
                preds_all.extend((prob[:, 1] > 0.5).long().cpu().numpy())

        return probs_all, preds_all
