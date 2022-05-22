import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Multihead_multifeats_QMixer(nn.Module):
    def __init__(self, args, scheme):
        super(Multihead_multifeats_QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_size = scheme["obs"]["vshape"]
        self.scheme = scheme

        self.embed_dim = args.mixing_embed_dim
        
        # multi-head module 2021/04/28   new try
        # self.input_size, self.num_heads, self.Seq_len, self.embedding_size = args
        # self.multi_head_module = Multihead_Module(args = (scheme["state"]["vshape"], self.obs_size, self.n_agents, self.n_agents, self.n_agents))
        # "move_feats":[agent_obs_feats[0]],
        # "enemy_feats":[agent_obs_feats[1]],
        # "ally_feats":[agent_obs_feats[2]],
        # "own_feats":[agent_obs_feats[3]]
        self.move_feats_head = Multihead_Module(args = (scheme["state"]["vshape"], scheme["move_feats_size"]["vshape"], self.n_agents, self.n_agents, self.n_agents))
        self.enemy_feats_head = Multihead_Module(args = (scheme["state"]["vshape"], scheme["enemy_feats_size"]["vshape"], self.n_agents, self.n_agents, self.n_agents))
        self.ally_feats_head = Multihead_Module(args = (scheme["state"]["vshape"], scheme["ally_feats_size"]["vshape"], self.n_agents, self.n_agents, self.n_agents))
        self.own_feats_head = Multihead_Module(args = (scheme["state"]["vshape"], scheme["own_feats_size"]["vshape"], self.n_agents, self.n_agents, self.n_agents))
        self.feats_attention_w = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )



        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def get_multi_head_info(self, state, obs_ls):
        obs_ls = obs_ls.reshape(-1, self.n_agents, self.obs_size)
        multi_head_weights = self.multi_head_module((state, obs_ls))
        return multi_head_weights

    def get_multi_head_multi_info(self, state, feats_ls):
        move_feats, enemy_feats, ally_feats, own_feats = feats_ls
        
        move_feats = move_feats.reshape(-1, self.n_agents, self.scheme["move_feats_size"]["vshape"]) 
        enemy_feats = enemy_feats.reshape(-1, self.n_agents, self.scheme["enemy_feats_size"]["vshape"])
        ally_feats = ally_feats.reshape(-1, self.n_agents, self.scheme["ally_feats_size"]["vshape"])
        own_feats = own_feats.reshape(-1, self.n_agents, self.scheme["own_feats_size"]["vshape"])
        
        move_weights = self.move_feats_head((state, move_feats)).unsqueeze(1)
        enemy_weights = self.enemy_feats_head((state, enemy_feats)).unsqueeze(1)
        ally_weights = self.ally_feats_head((state, ally_feats)).unsqueeze(1)
        own_weights = self.own_feats_head((state, own_feats)).unsqueeze(1)

        # feats_net_weights = th.abs(self.feats_attention_w(state).reshape(-1, self.state_dim, 4))
        # state = state.view(-1, 1, self.state_dim) 
        feats_weights = F.softmax(self.feats_attention_w(state), -1)
        feats_weights = feats_weights.unsqueeze(-1).unsqueeze(-1)
        feats_weights = feats_weights.repeat(1, 1, self.n_agents, self.n_agents)
        att_weights = (feats_weights * th.cat([move_weights, enemy_weights, ally_weights, own_weights], 1)).sum(1).squeeze(1)
        # att_weights = F.softmax(move_weights + enemy_weights + ally_weights + own_weights, -1)
        return att_weights



    def forward(self, agent_qs, states, feats_ls):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        multi_head_weights = self.get_multi_head_multi_info(states, feats_ls)
        agent_qs = th.bmm(multi_head_weights, agent_qs.permute(0, 2, 1)).permute(0, 2, 1)


        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot





class GATLayer(nn.Module):
    def __init__(self, args):
        super(GATLayer, self).__init__()
        self.in_features, self.out_features, self.dropout, self.alpha, self.concat = args

        self.W = nn.Parameter(th.zeros(size = (self.in_features, self.out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(th.zeros(size = (2*self.out_features, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input):
        inputs, Adj = input
        h = th.matmul(inputs, self.W)
        N = h.size()[-2]
        BATCH_SIZE = h.size()[0]
        SEQ_LEN = h.size()[1]

        # Attention mechanism
        a_input = th.cat([h.repeat(1, 1, 1, N).view(BATCH_SIZE, SEQ_LEN, N*N, -1), h.repeat(1, 1, N, 1)], -1).view(BATCH_SIZE, SEQ_LEN, N, -1, 2*self.out_features)
        e = self.leakyrelu(th.matmul(a_input, self.a).squeeze(-1))

        # mask attention
        zero_vec = -9e15 * th.ones_like(e)
        attention = th.where(Adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = th.matmul(attention, h)

        if self.concat:
            return (F.elu(h_prime), Adj)
        else:
            return h_prime



class GAT_Module(nn.Module):
    def __init__(self, args):
        super(GAT_Module, self).__init__()
        self.input_size, self.output_size, self.dropout, self.alpha = args
        self.gat_net = nn.Sequential(
            GATLayer(args = (self.input_size, 64, self.dropout, self.alpha, True)),
            GATLayer(args = (64, self.output_size, self.dropout, self.alpha, False))
        )

    def forward(self, inputs):
        return self.gat_net(inputs)


class Multihead_Module(nn.Module):
    def __init__(self, args):
        super(Multihead_Module, self).__init__()
        self.state_size, self.input_size, self.num_heads, self.Seq_len, self.embedding_size = args

        self.hyper_q = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.input_size * self.embedding_size)
        )

        self.hyper_k = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.input_size * self.embedding_size)
        )

        self.hyper_v = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.input_size * self.embedding_size)
        )

        self.multihead_net = nn.MultiheadAttention(self.embedding_size, self.num_heads)


    def forward(self, input):
        state, inputs = input
        self.weight_q = self.hyper_q(state).view(-1, self.input_size, self.embedding_size)
        self.weight_k = self.hyper_k(state).view(-1, self.input_size, self.embedding_size)
        self.weight_v = self.hyper_v(state).view(-1, self.input_size, self.embedding_size)
        q_vec = th.bmm(inputs, self.weight_q).permute(1, 0, 2)
        k_vec = th.bmm(inputs, self.weight_k).permute(1, 0, 2)
        v_vec = th.bmm(inputs, self.weight_v).permute(1, 0, 2)
        multihead_op, multihead_weights = self.multihead_net(q_vec, k_vec, v_vec)
        return multihead_weights


# 不用nn.multiheadattention 实现 multihead
class dot_attention(nn.Module):
    """ 点积注意力机制"""

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播
        :param q:
        :param k:
        :param v:
        :param scale:
        :param attn_mask:
        :return: 上下文张量和attention张量。
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale        # 是否设置缩放
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)     # 给需要mask的地方设置一个负无穷。
        # 计算softmax
        attention = F.softmax(attention, -1)
        # 添加dropout
        attention = self.dropout(attention)
        # 和v做点积。
        context = torch.bmm(attention, v)
        return context, attention