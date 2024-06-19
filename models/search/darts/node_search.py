import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from .operations import *
from .node_operations import *
from .genotypes import *
from .model_search import FusionMixedOp

# #Original
# class NodeCell(nn.Module):
#     def __init__(self, node_steps, node_multiplier, args):
#         super().__init__()
        
#         self.args = args
        
#         self.node_steps = node_steps
#         self.node_multiplier = node_multiplier
        
#         self.edge_ops = nn.ModuleList()
#         self.node_ops = nn.ModuleList()
        
#         self.C = args.C
#         self.L = args.L
        
#         self.num_input_nodes = 2
#         # self.num_keep_edges = 2

#         for i in range(self.node_steps):
#             for j in range(self.num_input_nodes+i):
#                 edge_op = FusionMixedOp(self.C, self.L, self.args)
#                 self.edge_ops.append(edge_op)
                
#         for i in range(self.node_steps):
#             node_op = NodeMixedOp(self.C, self.L, self.args)
#             self.node_ops.append(node_op)

#         if self.node_multiplier != 1:
#             self.out_conv = nn.Conv1d(self.C * self.node_multiplier, self.C, 1, 1)
#             self.bn = nn.BatchNorm1d(self.C)
#             self.out_dropout = nn.Dropout(args.drpt)

#         # skip v3 and v4
#         self.ln = nn.LayerNorm([self.C, self.L])
#         self.dropout = nn.Dropout(args.drpt)

#     def forward(self, x, y, edge_weights, node_weights):
#         states = [x, y]
#         # init_state = self.node_ops[0](x, y, node_weights[0])
#         # states.append(init_state)
#         offset = 0
#         for i in range(self.node_steps):
#             step_input_feature = sum(self.edge_ops[offset+j](h, edge_weights[offset+j]) for j, h in enumerate(states))
#             s = self.node_ops[i](step_input_feature, step_input_feature, node_weights[i])
#             offset += len(states)
#             states.append(s)

#         out = torch.cat(states[-self.node_multiplier:], dim=1)
#         if self.node_multiplier != 1:
#             out = self.out_conv(out)
#             out = self.bn(out)
#             out = F.relu(out)
#             out = self.out_dropout(out)
        
#         # skip v4
#         out += x
#         out = self.ln(out)
        
#         return out
    
#Weighted fusion
class NodeCell(nn.Module):
    def __init__(self, node_steps, node_multiplier, args):
        super().__init__()
        
        self.args = args
        self.node_steps = node_steps
        self.node_multiplier = node_multiplier
        
        self.edge_ops = nn.ModuleList()
        self.node_ops = nn.ModuleList()
        
        self.C = args.C
        self.L = args.L
        
        self.num_input_nodes = 2

        for i in range(self.node_steps):
            for j in range(self.num_input_nodes+i):
                edge_op = FusionMixedOp(self.C, self.L, self.args)
                self.edge_ops.append(edge_op)
                
        for i in range(self.node_steps):
            node_op = NodeMixedOp(self.C, self.L, self.args)
            self.node_ops.append(node_op)

        # Learnable weights for the fusion
        self.weights = nn.Parameter(torch.Tensor(self.node_steps))
        nn.init.uniform_(self.weights)

        if self.node_multiplier != 1:
            # Adjust the number of input channels to match the expected number
            self.out_conv = nn.Conv1d(self.C, self.C, 1, 1)  # Adjust input channels here
            self.bn = nn.BatchNorm1d(self.C)
            self.out_dropout = nn.Dropout(args.drpt)

        self.ln = nn.LayerNorm([self.C, self.L])
        self.dropout = nn.Dropout(args.drpt)

    def forward(self, x, y, edge_weights, node_weights):
        states = [x, y]
        offset = 0
        weighted_outputs = []  # List to store the weighted outputs of each step
        for i in range(self.node_steps):
            step_input_feature = sum(self.edge_ops[offset+j](h, edge_weights[offset+j]) for j, h in enumerate(states))
            s = self.node_ops[i](step_input_feature, step_input_feature, node_weights[i])
            
            offset += len(states)
            states.append(s)
            
            # Weight the output of the current step and append to the list
            weighted_outputs.append(self.weights[i] * s)

        # Perform a weighted sum of the outputs from different steps
        weighted_sum = sum(weighted_outputs)

        if self.node_multiplier != 1:
            weighted_sum = self.out_conv(weighted_sum)
            weighted_sum = self.bn(weighted_sum)
            weighted_sum = F.relu(weighted_sum)
            weighted_sum = self.out_dropout(weighted_sum)

        weighted_sum += x
        weighted_sum = self.ln(weighted_sum)

        return weighted_sum



#Channel attention based fusion
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.max_pool = nn.AdaptiveMaxPool1d(1)

#         self.fc = nn.Sequential(
#             nn.Conv1d(in_channels, in_channels // reduction_ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv1d(in_channels // reduction_ratio, in_channels, 1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         avg_out = self.avg_pool(x)
#         max_out = self.max_pool(x)
#         avg_out = self.fc(avg_out)
#         max_out = self.fc(max_out)
#         return avg_out * x + max_out * x

# class NodeCell(nn.Module):
#     def __init__(self, node_steps, node_multiplier, args):
#         super(NodeCell, self).__init__()

#         self.args = args
#         self.node_steps = node_steps
#         self.node_multiplier = node_multiplier

#         self.edge_ops = nn.ModuleList()
#         self.node_ops = nn.ModuleList()

#         self.C = args.C
#         self.L = args.L
#         self.num_input_nodes = 2

#         for i in range(self.node_steps):
#             for j in range(self.num_input_nodes+i):
#                 edge_op = FusionMixedOp(self.C, self.L, self.args)
#                 self.edge_ops.append(edge_op)

#         for i in range(self.node_steps):
#             node_op = NodeMixedOp(self.C, self.L, self.args)
#             self.node_ops.append(node_op)

#         # Channel attention module
#         self.channel_attention = ChannelAttention(self.C * self.node_multiplier)

#         if self.node_multiplier != 1:
#             self.out_conv = nn.Conv1d(self.C * self.node_multiplier, self.C, 1, 1)
#             self.bn = nn.BatchNorm1d(self.C)
#             self.out_dropout = nn.Dropout(args.drpt)

#         self.ln = nn.LayerNorm([self.C, self.L])
#         self.dropout = nn.Dropout(args.drpt)

#     def forward(self, x, y, edge_weights, node_weights):
#         states = [x, y]
#         offset = 0
#         outputs = []  # List to store the outputs of each step

#         for i in range(self.node_steps):
#             step_input_feature = sum(self.edge_ops[offset+j](h, edge_weights[offset+j]) for j, h in enumerate(states))
#             s = self.node_ops[i](step_input_feature, step_input_feature, node_weights[i])
            
#             offset += len(states)
#             states.append(s)
            
#             # Append the output of the current step
#             outputs.append(s)

#         # Concatenate the outputs from different steps
#         concatenated_output = torch.cat(outputs[-self.node_multiplier:], dim=1)

#         # Apply channel attention
#         attended_output = self.channel_attention(concatenated_output)

#         if self.node_multiplier != 1:
#             attended_output = self.out_conv(attended_output)
#             attended_output = self.bn(attended_output)
#             attended_output = F.relu(attended_output)
#             attended_output = self.out_dropout(attended_output)

#         attended_output += x
#         attended_output = self.ln(attended_output)

#         return attended_output

















class FusionNode(nn.Module):
    
    def __init__(self, node_steps, node_multiplier, args):
        super().__init__()
        # self.logger = logger
        self.node_steps = node_steps
        self.node_multiplier = node_multiplier
        self.node_cell = NodeCell(node_steps, node_multiplier, args)

        self.num_input_nodes = 2
        self.num_keep_edges = 2
        
        self._initialize_betas()
        self._initialize_gammas()
        #self.count = 0
        self._arch_parameters = [self.betas, self.gammas]
        
    def _initialize_betas(self):
        k = sum(1 for i in range(self.node_steps) for n in range(self.num_input_nodes+i))
        num_ops = len(STEP_EDGE_PRIMITIVES)
        # beta controls node cell arch
        self.betas = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
    
    def compute_arch_entropy_gamma(self, dim=-1):
        alpha = self.arch_parameters()[0]
        #print(alpha.shape)
        prob = F.softmax(alpha, dim=dim)
        log_prob = F.log_softmax(alpha, dim=dim)
        entropy = - (log_prob * prob).sum(-1, keepdim=False)
        return entropy

    def _initialize_gammas(self):
        k = sum(1 for i in range(self.node_steps))
        num_ops = len(STEP_STEP_PRIMITIVES)
        #print("No of step_step_primitives:", len(STEP_STEP_PRIMITIVES))
        # gamma controls node_step_nodes arch
        self.gammas = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
        #print("self_gamma shape:",self.gammas.shape)
    
    def gumbel_softmax_sample(self, logits, temperature):
            """
            Sample from Gumbel-Softmax distribution.
            """
            gumbel_noise = torch.empty_like(logits).uniform_(0, 1)
            gumbel_noise = -torch.log(-torch.log(gumbel_noise + 1e-20) + 1e-20)  # Gumbel noise
            logits = (logits + gumbel_noise) / temperature
            #print("**********logits:",logits.shape)
            return F.softmax(logits, dim=-1)

    def gumbel_softmax_sample_node(self, logits,num_sample ,temperature):
            """
            Sample from Gumbel-Softmax distribution.
            """
            sample = []
            for i in range(num_sample):
                gumbel_noise2 = torch.empty_like(logits).uniform_(0, 1)
                gumbel_noise2 = -torch.log(-torch.log(gumbel_noise2 + 1e-20) + 1e-20)  # Gumbel noise
                logits = (logits + gumbel_noise2) / temperature
                sample_index = torch.argmax(logits,dim=1)
                for i in sample_index:
                    sample.append(STEP_STEP_PRIMITIVES[i])

            return F.softmax(logits, dim=-1)
    
    
    def forward(self, x, y):
        edge_weights=self.gumbel_softmax_sample(self.betas, temperature=10.0)
        node_weights=self.gumbel_softmax_sample_node(self.gammas,num_sample = 15,temperature=10.0)
        #print("###################### Node Logits: ", node_logits.shape)
        
        #edge_weights = F.softmax(self.betas, dim=-1)
        #node_weights = F.softmax(self.gammas, dim=-1)
        #print(STEP_STEP_PRIMITIVES)
        #print("************Node weights:",node_weights.shape)
        #self.count+=1
        #print("self-count: ",self.count)
        out = self.node_cell(x, y, edge_weights, node_weights)

        #print(out.shape)     
        return out

    def arch_parameters(self):  
        return self._arch_parameters

    def node_genotype(self):
        def _parse(edge_weights, node_weights):
            edge_gene = []
            node_gene = []

            n = 2
            start = 0
            for i in range(self.node_steps):
                end = start + n
                
                W = edge_weights[start:end]
                edges = sorted(range(i + self.num_input_nodes), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:self.num_keep_edges]
                
                # print("edges:", edges)
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != STEP_EDGE_PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    edge_gene.append((STEP_EDGE_PRIMITIVES[k_best], j))
                    # gene.append((PRIMITIVES[k_second_best], j))

                start = end
                n += 1
                
            for i in range(self.node_steps):
                W = node_weights[i]
                k_best = None
                for k in range(len(W)):
                    if k_best is None or W[k] > W[k_best]:
                        k_best = k

                node_gene.append((STEP_STEP_PRIMITIVES[k_best]))

            return edge_gene, node_gene

        concat_gene = range(self.num_input_nodes+self.node_steps-self.node_multiplier, self.node_steps+self.num_input_nodes)
        concat_gene = list(concat_gene)

        edge_weights = F.softmax(self.betas, dim=-1)
        node_weights = F.softmax(self.gammas, dim=-1)
        
        edge_gene, node_gene = _parse(edge_weights, node_weights)

        fusion_gene = StepGenotype(
            inner_edges = edge_gene,
            inner_steps = node_gene,
            inner_concat = concat_gene,
        )
        # print(concat_gene)
        # print(edge_gene)
        # print(node_gene)
        return fusion_gene

if __name__ == '__main__':
    class Args():
        def __init__(self, C, L):
            self.C = C
            self.L = L
            self.drpt = 0.1

    args = Args(16, 8)
    node_cell = NodeCell(2, 1, args)
    fusion_node = FusionNode(2, 1, args)

    a = torch.randn(4, 16, 8)
    b = torch.randn(4, 16, 8)
    # cat_conv_glu = CatConvGlu(16, 8)
    # cat_conv_relu = CatConvRelu(16, 8)

    fusion_node(a, b).shape
    fusion_node.gammas.shape
    fusion_node.node_genotype()







