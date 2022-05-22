import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Graph_Mixer(nn.Module):
    def __init__(self, args, scheme):
        super(Graph_Mixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_size = scheme["obs"]["vshape"]

        self.embed_dim = args.mixing_embed_dim
        
        # graph assist   self.input_size, self.output_size, self.dropout, self.alpha, self.device = args
        self.gat_module = GAT_Module(args = (scheme["state"]["vshape"], scheme["obs"]["vshape"], 1, 0.6, 0.2))


    def get_graph_info(self, states, obs_ls, adj_ls):
        gat_op = self.gat_module((states, obs_ls, adj_ls)).view(-1, self.n_agents, 1)
        gat_weights = F.softmax(gat_op, dim=0)
        return gat_weights


    def forward(self, agent_qs, states, obs_ls, adj_ls):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        graph_weights = self.get_graph_info(obs_ls, adj_ls)
        q_tot = th.bmm(agent_qs, graph_weights).squeeze(-1)

        return q_tot





class GATLayer(nn.Module):
    def __init__(self, args):
        super(GATLayer, self).__init__()
        self.hyper_input_size ,self.in_features, self.out_features, self.dropout, self.alpha, self.concat = args

        self.hyper_w = nn.Sequential(nn.Linear(self.hyper_input_size, 32),
                                           nn.ReLU(),
                                           nn.Linear(32, self.out_features * self.in_features))

        self.hyper_a = nn.Sequential(nn.Linear(self.state_size, 32),
                                            nn.ReLU(),
                                            nn.Linear(32, self.out_features * 2))
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def get_weights(self, inputs):
        self.W = th.abs(self.hyper_w(inputs)).view(-1, self.in_features, self.out_features)
        self.a = th.abs(self.hyper_a(inputs)).view(-1, 2*self.out_features, 1)


    def forward(self, input):
        hyper_inputs, inputs, Adj = input
        self.get_weights(hyper_inputs)
        self.W = self.W.view(inputs.size()[0], inputs.size()[1], self.in_features, self.out_features)
        self.a = self.a.view(inputs.size()[0], inputs.size()[1], 2*self.out_features, 1)
        h = th.matmul(inputs, self.W)
        N = h.size()[-2]
        BATCH_SIZE = h.size()[0]
        SEQ_LEN = h.size()[1]
        self.a = self.a.repeat(1,1,N,1).view(inputs.size()[0], inputs.size()[1], N, 2*self.out_features, 1)
        
        # Attention mechanism
        a_input = th.cat([h.repeat(1, 1, 1, N).view(BATCH_SIZE, SEQ_LEN, N*N, -1), h.repeat(1, 1, N, 1)], -1).view(BATCH_SIZE, SEQ_LEN, N, -1, 2*self.out_features)
        e = self.leakyrelu(th.matmul(a_input, self.a).squeeze(-1))

        # mask attention
        zero_vec = -9e15 * th.ones_like(e)
        attention = th.where(Adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        if self.dropout != 0:
            attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = th.matmul(attention, h)

        if self.concat:
            return (hyper_inputs, F.elu(h_prime), Adj)
        else:
            return h_prime



class GAT_Module(nn.Module):
    def __init__(self, args):
        super(GAT_Module, self).__init__()
        self.hyper_size, self.input_size, self.output_size, self.dropout, self.alpha = args
        self.gat_net = nn.Sequential(
            GATLayer(args = (self.hyper_size, self.input_size, 64, self.dropout, self.alpha, True)),
            GATLayer(args = (self.hyper_size, 64, self.output_size, 0, self.alpha, False))
        )

    def forward(self, inputs):
        return self.gat_net(inputs)

if __name__ == "__main__":
    a = th.randn(32, 60, 5, 24)
    hyper_inputs = th.randn(32, 60, 120)
    adj = th.ones(a.shape[0], a.shape[1], a.shape[2], a.shape[2])
    model = GAT_Module(args = (120, 24, 5, 0.6, 0.2))
    c = model((hyper_inputs, a, adj))
    print(c.shape)