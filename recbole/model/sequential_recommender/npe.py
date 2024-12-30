# -*- coding: utf-8 -*-
# @Time     : 2020/11/20 22:33
# @Author   : Shao Weiqi
# @Reviewer : Lin Kun
# @Email    : shaoweiqi@ruc.edu.cn

r"""
SHAN
################################################

Reference:
    Ying, H et al. "Sequential Recommender System based on Hierarchical Attention Network."in IJCAI 2018


"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_, uniform_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer


class NPE(SequentialRecommender):
    r"""
    SHAN exploit the Hierarchical Attention Network to get the long-short term preference
    first get the long term purpose and then fuse the long-term with recent items to get long-short term purpose

    """

    def __init__(self, config, dataset):

        super(NPE, self).__init__(config, dataset)

        # load the dataset information
        self.n_users = dataset.num(self.USER_ID)
        self.device = config['device']

        # load the parameter information
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.short_item_length = config["short_item_length"]  # the length of the short session items
        assert self.short_item_length <= self.max_seq_length, "short_item_length can't longer than the max_seq_length"
        self.reg_weight = config["reg_weight"]
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.selected_features = config['selected_features']
        self.num_feature_field = len(config['selected_features'])
        self.pooling_mode = config['pooling_mode']
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.user_embedding = nn.Embedding(self.n_users, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        # 全连接层
        self.long_w = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.long_b = nn.Parameter(
            uniform_(
                tensor=torch.zeros(self.hidden_size),
                a=-np.sqrt(3 / self.hidden_size),
                b=np.sqrt(3 / self.hidden_size)
            ),
            requires_grad=True
        ).to(self.device)
        self.long_short_w = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.long_short_b = nn.Parameter(
            uniform_(
                tensor=torch.zeros(self.hidden_size),
                a=-np.sqrt(3 / self.hidden_size),
                b=np.sqrt(3 / self.hidden_size)
            ),
            requires_grad=True
        ).to(self.device)

        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset, self.hidden_size, self.selected_features, self.pooling_mode, self.device
        )

        # self.item_trm_encoder = TransformerEncoder(
        #     n_layers=self.n_layers,
        #     n_heads=self.n_heads,
        #     hidden_size=self.hidden_size,
        #     inner_size=self.inner_size,
        #     hidden_dropout_prob=self.hidden_dropout_prob,
        #     attn_dropout_prob=self.attn_dropout_prob,
        #     hidden_act=self.hidden_act,
        #     layer_norm_eps=self.layer_norm_eps
        # )
        # self.feature_trm_encoder = TransformerEncoder(
        #     n_layers=self.n_layers,
        #     n_heads=self.n_heads,
        #     hidden_size=self.hidden_size,
        #     inner_size=self.inner_size,
        #     hidden_dropout_prob=self.hidden_dropout_prob,
        #     attn_dropout_prob=self.attn_dropout_prob,
        #     hidden_act=self.hidden_act,
        #     layer_norm_eps=self.layer_norm_eps
        # )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.relu = nn.ReLU()

        self.loss_type = config['loss_type']
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # init the parameter of the model
        self.apply(self.init_weights)
        self.item_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=self.attn_dropout_prob,
            bidirectional=True
        )
        self.feature_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=self.attn_dropout_prob,
            bidirectional=True
        )

    def reg_loss(self, user_embedding, item_embedding):

        reg_1, reg_2 = self.reg_weight
        loss_1 = reg_1 * torch.norm(self.long_w.weight, p=2) + reg_1 * torch.norm(self.long_short_w.weight, p=2) \
                 + reg_1 * torch.norm(self.concat_layer.weight, p=2)
        loss_2 = reg_2 * torch.norm(user_embedding, p=2) + reg_2 * torch.norm(item_embedding, p=2)

        return loss_1 + loss_2

    def inverse_seq_item(self, seq_item, seq_item_len):
        """
        inverse the seq_item, like this
            [1,2,3,0,0,0,0] -- after inverse -->> [0,0,0,0,1,2,3]
        """
        seq_item = seq_item.cpu().numpy()
        seq_item_len = seq_item_len.cpu().numpy()
        new_seq_item = []
        for items, length in zip(seq_item, seq_item_len):
            item = list(items[:length])
            zeros = list(items[length:])
            seqs = zeros + item
            new_seq_item.append(seqs)
        seq_item = torch.tensor(new_seq_item, dtype=torch.long, device=self.device)

        return seq_item
    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, seq_item, user, seq_item_len):

        seq_item = self.inverse_seq_item(seq_item, seq_item_len)

        position_ids = torch.arange(seq_item.size(1), dtype=torch.long, device=seq_item.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seq_item)
        position_embedding = self.position_embedding(position_ids)

        seq_item_embedding = self.item_embedding(seq_item) + position_embedding
        user_embedding = self.user_embedding(user)

        item_output,_ = self.item_lstm(seq_item_embedding)


        sparse_embedding, dense_embedding = self.feature_embed_layer(None, seq_item)
        sparse_embedding = sparse_embedding['item']
        dense_embedding = dense_embedding['item']
        # concat the sparse embedding and float embedding
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)
        feature_emb = torch.cat(feature_table, dim=-2).squeeze(2)
        feature_emb = feature_emb + position_embedding
        feature_output,_ = self.feature_lstm(feature_emb)


        # extended_attention_mask = self.get_attention_mask(seq_item)
        # item_trm_output = self.item_trm_encoder(seq_item_embedding, extended_attention_mask,
        #                                         output_all_encoded_layers=True)
        # item_output = item_trm_output[-1]
        #
        # feature_trm_output = self.feature_trm_encoder(
        #     feature_emb, extended_attention_mask, output_all_encoded_layers=True
        # )
        # feature_output = feature_trm_output[-1]

        # get the mask
        mask = seq_item.data.eq(0)
        long_term_item_pooling_layer = self.long_term_item_pooling_layer(item_output, user_embedding, mask)
        # batch_size * 1 * embedding_size

        long_term_attr_pooling_layer = self.long_term_item_pooling_layer(feature_output, user_embedding, mask)

        short_item_embedding = item_output[:, -self.short_item_length:, :]
        mask_long_short = mask[:, -self.short_item_length:]
        batch_size = mask_long_short.size(0)
        x = torch.zeros(size=(batch_size, 1)).eq(1).to(self.device)
        mask_long_short = torch.cat([x, mask_long_short], dim=1)
        # batch_size * short_item_length * embedding_size
        long_short_item_embedding = torch.cat([long_term_item_pooling_layer, short_item_embedding], dim=1)
        # batch_size * 1_plus_short_item_length * embedding_size

        long_short_item_embedding = self.long_and_short_term_attention_based_pooling_layer(
            long_short_item_embedding, user_embedding, mask_long_short
        )
        short_feature_embedding = feature_output[:, -self.num_feature_field*self.short_item_length:, :]
        long_shor_feature_embedding = torch.cat([long_term_attr_pooling_layer,short_feature_embedding], dim=1)
        long_short_feature_embedding = self.long_and_short_term_attention_based_pooling_layer(
            long_shor_feature_embedding,user_embedding, mask_long_short
        )

        # item_output = self.gather_indexes(item_output, seq_item_len - 1)  # [B H]
        # feature_output = self.gather_indexes(feature_output, seq_item_len - 1)  # [B H]
        output_concat = torch.cat((long_short_item_embedding, long_short_feature_embedding), -1)  # [B 2*H]
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        #seq_output = self.dropout(output)
        # batch_size * embedding_size
        return output

    def calculate_loss(self, interaction):

        seq_item = interaction[self.ITEM_SEQ]
        seq_item_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        user_embedding = self.user_embedding(user)
        seq_output = self.forward(seq_item, user, seq_item_len)
        pos_items = interaction[self.POS_ITEM_ID]
        pos_items_emb = self.item_embedding(pos_items)
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss + self.reg_loss(user_embedding, pos_items_emb)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss + self.reg_loss(user_embedding, pos_items_emb)

    def predict(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        test_item = interaction[self.ITEM_ID]
        seq_item_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, user, seq_item_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        seq_item_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, user, seq_item_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores

    def long_and_short_term_attention_based_pooling_layer(self, long_short_item_embedding, user_embedding, mask=None):
        """

        fusing the long term purpose with the short-term preference
        """
        long_short_item_embedding_value = long_short_item_embedding

        long_short_item_embedding = self.sigmoid(self.long_short_w(long_short_item_embedding) + self.long_short_b)
        long_short_item_embedding = torch.matmul(long_short_item_embedding, user_embedding.unsqueeze(2)).squeeze(-1)
        # batch_size * seq_len
        if mask is not None:
            long_short_item_embedding.masked_fill_(mask, -1e9)
        long_short_item_embedding = nn.Softmax(dim=-1)(long_short_item_embedding)
        long_short_item_embedding = torch.mul(long_short_item_embedding_value,
                                              long_short_item_embedding.unsqueeze(2)).sum(dim=1)
        return long_short_item_embedding

    def long_term_item_pooling_layer(self, seq_item_embedding, user_embedding, mask=None):
        """

        get the long term purpose of user
        """
        seq_item_embedding_value = seq_item_embedding
        user_embedding = user_embedding.unsqueeze(2)
        seq_item_embedding = self.sigmoid(self.long_w(seq_item_embedding) + self.long_b)
        user_item_embedding = torch.matmul(seq_item_embedding, user_embedding).squeeze(-1)
        # batch_size * seq_len
        if mask is not None:
            user_item_embedding.masked_fill_(mask, -1e9)
        user_item_embedding = nn.Softmax(dim=1)(user_item_embedding)
        user_item_embedding = torch.mul(seq_item_embedding_value,
                                        user_item_embedding.unsqueeze(2)).sum(dim=1, keepdim=True)
        # batch_size * 1 * embedding_size

        return user_item_embedding


