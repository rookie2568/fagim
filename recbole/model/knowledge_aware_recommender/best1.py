# -*- coding: utf-8 -*-
# @Time     : 2020/11/20 22:33
# @Author   : Shao Weiqi
# @Reviewer : Lin Kun
# @Email    : shaoweiqi@ruc.edu.cn

r"""

################################################

Reference:
    Ying, H et al. "Sequential Recommender System based on Hierarchical Attention Network."in IJCAI 2018


"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, uniform_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)


class BEST(SequentialRecommender):
    r"""
    SHAN exploit the Hierarchical Attention Network to get the long-short term preference
    first get the long term purpose and then fuse the long-term with recent items to get long-short term purpose

    """

    def __init__(self, config, dataset):

        super(BEST, self).__init__(config, dataset)

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

        self.item_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.concat_layer = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.relu = nn.ReLU()
        self.caps_num = 5
        self.caps_linear = nn.Linear(16, self.hidden_size)
        self.k = 2
        # 胶囊网络
        self.caps_net = CapsNet()
        self.loss_type = config['loss_type']
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # init the parameter of the model
        self.apply(self.init_weights)
        self.other_parameter_name = ['feature_embed_layer']
        self.batch = nn.BatchNorm1d(num_features=self.caps_num)
        self.temperature_para = 0.5
        self.multi_layer = nn.Linear(self.hidden_size,1)
        self.k_to_hidden = nn.Linear(self.k, self.hidden_size)

    def reg_loss(self, user_embedding, item_embedding):

        reg_1, reg_2 = self.reg_weight
        loss_1 = reg_1 * torch.norm(self.long_w.weight, p=2) + reg_1 * torch.norm(self.long_short_w.weight, p=2)
        + reg_1 * torch.norm(self.concat_layer.weight, p=2) + reg_1 * torch.norm(self.caps_linear.weight, p=2)
        loss_2 = reg_2 * torch.norm(user_embedding, p=1) + reg_2 * torch.norm(item_embedding, p=1)

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
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight.data)
        elif isinstance(module, nn.BatchNorm1d):  # 这里建议去学习一下BN的知识，有空我也会再写一篇
            module.weight.data.fill_(1)
            module.bias.data.zero_()

    def forward(self, seq_item, user, seq_item_len):

        seq_item = self.inverse_seq_item(seq_item, seq_item_len)
        user_embedding = self.user_embedding(user)
        position_ids = torch.arange(seq_item.size(1), dtype=torch.long, device=seq_item.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seq_item)
        position_embedding = self.position_embedding(position_ids)

        seq_item_embedding = self.item_embedding(seq_item)
        seq_item_embedding = self.dropout(self.LayerNorm(seq_item_embedding)) + position_embedding

        caps_seq_embedding = self.caps_net(seq_item_embedding)
        caps_seq_embedding = self.batch(caps_seq_embedding)
        caps_seq_embedding = self.caps_linear(caps_seq_embedding)

        # get the mask
        mask = seq_item.data.eq(0)
        long_term_item_pooling_layer = self.long_term_item_pooling_layer(seq_item_embedding, user_embedding, mask)

        short_item_embedding = seq_item_embedding[:, -self.short_item_length:, :]
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

        extended_attention_mask = self.get_attention_mask(seq_item)
        item_trm_output = self.item_trm_encoder(seq_item_embedding, extended_attention_mask,
                                                output_all_encoded_layers=True)
        item_output = item_trm_output[-1]
        output = self.gather_indexes(item_output, seq_item_len - 1)
        output = self.LayerNorm(output)

        user_history_flat = seq_item_embedding.view(-1, seq_item_embedding.size(-1))
        interest_capsules_flat = caps_seq_embedding.view(-1, caps_seq_embedding.size(-1))

        # 计算点积相似度
        # 形状：[batch_size * seq_len, num_interests]
        similarity_scores = torch.matmul(user_history_flat, interest_capsules_flat.t())

        # 对相似度进行降序排序，选择top k个兴趣胶囊
        top_k_scores, top_k_indices = similarity_scores.topk(self.k, dim=1)

        # 将top k兴趣胶囊的索引恢复到原始形状
        # 形状：[batch_size, k, hidden_size]
        top_k_interests = caps_seq_embedding.view(caps_seq_embedding.size(0), -1, caps_seq_embedding.size(-1))
        top_k_interests = top_k_interests.gather(dim=1, index=top_k_indices.unsqueeze(-1).expand(-1, -1,
                                                                                                 caps_seq_embedding.size(
                                                                                                     -1)))
        interest_embed = self.sigmoid(top_k_interests)
        sigs = nn.Softmax(dim=1)(top_k_scores).unsqueeze(2).repeat(1, 1, self.hidden_size)
        interest_embed = interest_embed.mul(sigs).sum(dim=1)
        # interest_embed = self.multi_layer(interest_embed).squeeze(-1)
        # interest_embed = self.k_to_hidden(interest_embed)

        # interest_score = torch.matmul(self.item_embedding.weight, caps_seq_embedding.unsqueeze(2))
        # idx = interest_score.argsort(1)[:, -self.k:]
        # idx_si = interest_score.sort(1)[0][:, -self.k:]
        # interest_embed = torch.sigmoid(self.item_embedding(idx).squeeze(2))
        # sigs = nn.Softmax(dim=1)(idx_si).repeat(1, 1, self.hidden_size)
        # interest_embed = interest_embed.mul(sigs).sum(dim=1)
        # interest_embed = self.multi_layer(interest_embed).squeeze(-1)
        # interest_embed = self.k_to_hidden(interest_embed)

        interest_concat = torch.cat((interest_embed, output,long_short_item_embedding), -1)
        final = self.concat_layer(interest_concat)
        final = self.LayerNorm(final)

        return final

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
        long_short_item_embedding = nn.Softmax(dim=-1)(long_short_item_embedding/self.temperature_para)
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
        user_item_embedding = nn.Softmax(dim=1)(user_item_embedding/self.temperature_para)
        user_item_embedding = torch.mul(seq_item_embedding_value,
                                        user_item_embedding.unsqueeze(2)).sum(dim=1, keepdim=True)
        # batch_size * 1 * embedding_size

        return user_item_embedding


class CapsNet(nn.Module):
    """Basic implementation of capsule network layer."""

    def __init__(self):
        super(CapsNet, self).__init__()

        # Conv2d layer
        self.conv = nn.Conv1d(128, 50, 9)
        self.relu = nn.ReLU(inplace=True)

        # Primary capsule
        self.primary_caps = PrimaryCaps(num_conv_units=8,
                                        in_channels=42,
                                        out_channels=8,
                                        kernel_size=3,
                                        stride=1)

        # Digit capsule
        self.digit_caps = DigitCaps(in_dim=8,
                                    in_caps=384,
                                    out_caps=5,
                                    out_dim=16,
                                    num_routing=3)

        # Reconstruction layer
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid())

    def forward(self, x):
        out = self.relu(self.conv(x.permute(0, 2, 1)).permute(0, 2, 1))
        out = self.primary_caps(out)
        out = self.digit_caps(out)

        # Shape of logits: (batch_size, out_capsules)
        # logits = torch.norm(out, dim=-1)

        return out


class PrimaryCaps(nn.Module):
    """Primary capsule layer."""

    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        # Each conv unit stands for a single capsule.
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels * num_conv_units,
                              kernel_size=kernel_size,
                              stride=stride)
        self.out_channels = out_channels

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: out_capsules * (batch_size, out_channels, height, weight)
        out = self.conv(x)
        # Flatten out: (batch_size, out_capsules * height * weight, out_channels)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)


class DigitCaps(nn.Module):
    """Digit capsule layer."""

    def __init__(self, in_dim, in_caps, out_caps, out_dim, num_routing):
        """
            Initialize the layer.

            Args:
                in_dim: 		Dimensionality of each capsule vector.
                in_caps: 		Number of input capsules if digits layer.
                out_caps: 		Number of capsules in the capsule layer
                out_dim: 		Dimensionality, of the output capsule vector.
                num_routing:	Number of iterations during routing algorithm
            """
        super(DigitCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.num_routing = num_routing
        self.device = device
        self.W = nn.Parameter(0.01 * torch.randn(1, out_caps, in_caps, out_dim, in_dim),
                              requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        # W @ x =
        # (1, out_caps, in_caps, out_dim, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, out_caps, in_caps, out_dims, 1)
        u_hat = torch.matmul(self.W, x)
        # (batch_size, out_caps, in_caps, out_dim)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.out_caps, self.in_caps, 1).to(self.device)

        for route_iter in range(self.num_routing - 1):
            # (batch_size, out_caps, in_caps, 1) -> Softmax along out_caps
            c = b.softmax(dim=1)

            # element-wise multiplication
            # (batch_size, out_caps, in_caps, 1) * (batch_size, in_caps, out_caps, out_dim) ->
            # (batch_size, out_caps, in_caps, out_dim) sum across in_caps ->
            # (batch_size, out_caps, out_dim)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along out_dim
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, out_caps, in_caps, out_dim) @ (batch_size, out_caps, out_dim, 1)
            # -> (batch_size, out_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along out_dim
        v = squash(s)

        return v