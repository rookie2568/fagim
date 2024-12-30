

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer, VanillaAttention
from recbole.model.loss import BPRLoss
import torch.nn.functional as F


class GARec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(GARec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.num_feature_field = len(config['selected_features'])
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset, self.hidden_size, self.selected_features, self.pooling_mode, self.device
        )
        self.feature_att_layer = VanillaAttention(self.hidden_size, self.hidden_size)
        self.feature_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.trm_encoder = TransformerEncoder(
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
        self.num_groups = 7
        self.user_group_emb = nn.Embedding(self.num_groups, config["hidden_size"])
        self.ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.ffn2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.query_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation_fn = F.relu
        self.concat_layer = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.concat_layer2=nn.Linear(self.hidden_size*2,self.hidden_size)
        self.concat_layer3=nn.Linear(self.hidden_size*3,self.hidden_size)
        self.sigmoid=nn.Sigmoid()
        self.feature_w=nn.Linear(self.hidden_size,self.hidden_size)
        self.group_w = nn.Linear(self.hidden_size, self.hidden_size)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
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

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding['item']
        dense_embedding = dense_embedding['item']
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)
        feature_table = torch.cat(feature_table, dim=-2)
        feature_emb, attn_weight = self.feature_att_layer(feature_table)
        # feature position add position embedding
        feature_emb = feature_emb + position_embedding
        feature_emb = self.LayerNorm(feature_emb)
        feature_trm_input = self.dropout(feature_emb)



        extended_attention_mask = self.get_attention_mask(item_seq)
        feature_trm_output = self.feature_trm_encoder(
            feature_trm_input, extended_attention_mask, output_all_encoded_layers=True
        )  # [B Len H]
        feature_output = feature_trm_output[-1]
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        #256, 50, 64
        output = trm_output[-1]
        with torch.no_grad():
            seq_group = self.multi_head_attention(self.user_group_emb.weight, output,
                                                  num_units=self.hidden_size,
                                                  num_heads=self.n_heads,
                                                  dropout_rate=self.attn_dropout_prob, is_training=True,
                                                  causality=False)
            seq_group = self.ffn(seq_group)
            weigthed_group, attn_weight = self.group_attention(output, seq_group,
                                                               num_units=self.hidden_size,
                                                               num_groups=self.num_groups)
            weigthed_group = self.ffn2(weigthed_group)
            weigthed_group=torch.mean(weigthed_group,1)
        #256,64
        output = self.gather_indexes(output, item_seq_len - 1)
        feature_output = self.gather_indexes(feature_output, item_seq_len - 1)
        # output=output+self.weight_1*weigthed_group
        # output=torch.cat([output,weigthed_group,feature_output],dim=-1)
        output=torch.cat([output,weigthed_group],dim=-1)
        output=self.concat_layer(output)
        # output=torch.cat([output,feature_output],dim=-1)
        # output=self.concat_layer2(output)
        output = self.LayerNorm(output)
        output = self.dropout(output)
        # output = self.multi_source_embedding_fusion(output,feature_output,weigthed_group)
        return output  # [B H]

    def multi_source_embedding_fusion(self, item_emb, feature_emb,group_emb):
        hidden_emb=self.attention_based_item_feature_pooling(item_emb,feature_emb,item_seq_len=None)
        hidden_emb=self.LayerNorm(hidden_emb)
        hidden_emb=self.dropout(hidden_emb)
        hidden_emb2=self.attention_based_item_group_pooling(item_emb,group_emb,item_seq_len=None)
        hidden_emb2=self.LayerNorm(hidden_emb2)
        hidden_emb2=self.dropout(hidden_emb2)
        final=self.concat_layer([hidden_emb,hidden_emb2],dim=-1)
        return final

    def attention_based_item_feature_pooling(self, item_emb, feature_emb, item_seq_len):
        feature_embedding_value = feature_emb
        feature_embedding=self.sigmoid(self.feature_w(feature_embedding_value))
        item_feature_embedding = torch.matmul(feature_embedding, item_emb.unsqueeze(2)).squeeze(-1)
        return item_feature_embedding

    def attention_based_item_group_pooling(self, item_emb, group_emb, item_seq_len):
        group_embedding_value = group_emb
        group_embedding = self.sigmoid(self.group_w(group_embedding_value))
        item_group_embedding = torch.matmul(group_embedding, item_emb.unsqueeze(2)).squeeze(-1)
        return item_group_embedding

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)

        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight    # [n_items, H]

            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding(test_item)

        # [B H] * [H 1] = [B 1]
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

    def multi_head_attention(self, queries, keys,
                             num_units=None,
                             num_heads=8,
                             dropout_rate=0,
                             is_training=True,
                             causality=False,
                             ):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        # Linear projections
        queries = torch.unsqueeze(queries, 0)
        queries = queries.expand(keys.size(0), -1, -1)
        Q = self.query_layer(queries)
        K = self.key_layer(keys)
        V = self.value_layer(keys)
        # Split and concat
        Q_ = torch.cat(torch.split(Q, num_units // num_heads, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, num_units // num_heads, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, num_units // num_heads, dim=2), dim=0)
        # Multiplication
        outputs = torch.matmul(Q_, K_.transpose(-1, -2))
        # Scale
        outputs = outputs / (K_.size(-1) ** 0.5)
        # Key Masking
        key_masks = torch.sign(torch.abs(torch.sum(keys, axis=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(num_heads, 1)
        key_masks = key_masks.unsqueeze(1).repeat(1, queries.size(1), 1)

        paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
        # Causality = Future blinding
        if causality:
            diag_vals = torch.ones_like(outputs[0, :, :])
            tril = torch.tril(diag_vals)
            masks = tril.unsqueeze(0).repeat(outputs.size(0), 1, 1)

            paddings = torch.ones_like(masks) * (-2 ** 32 + 1)
            outputs = torch.where(torch.eq(masks, 0), paddings, outputs)
        # Activation
        outputs = F.softmax(outputs, dim=-1)
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, axis=-1)))
        query_masks = query_masks.repeat(num_heads, 1)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, keys.size(1))
        outputs *= query_masks
        # Dropouts
        outputs = F.dropout(outputs, training=is_training, p=dropout_rate)
        # Weighted sum
        outputs = torch.matmul(outputs, V_)
        # Restore shape
        outputs = torch.cat(torch.split(outputs, outputs.size(0) // num_heads, dim=0), dim=2)
        # Residual connection
        outputs += queries
        # Normalize
        outputs = self.LayerNorm(outputs)
        return outputs

    def  group_attention(self,queries,
                            keys,
                            num_units=None,
                            num_groups=None):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        queries = queries.transpose(1, 2)#E
        q_projection=nn.Linear(queries.size(-1),num_groups,bias=False).to(queries.device)
        # q_projection=nn.Linear(queries.size(-1),num_units,bias=False).to(queries.device)
        v_projection=nn.Linear(keys.size(-1),num_units,bias=False).to(queries.device)
        queries=q_projection(queries)
        queries=queries.transpose(1,2)

        v=v_projection(keys)#G
        outputs=queries
        layer_sizes=[1]
        for idx, layer_size in enumerate(layer_sizes):
            curr_w_nn_layer = nn.Linear(outputs.size(-1), layer_size).to(queries.device)
            outputs = curr_w_nn_layer(outputs)
            outputs = self.activation_fn(outputs)
        nn_output = outputs
        attn_weight=F.softmax(nn_output,dim=1)
        outputs=torch.multiply(attn_weight,v)
        return outputs,attn_weight

