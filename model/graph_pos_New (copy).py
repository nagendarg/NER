
from typing import *
import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from data_utils import documents
#from losses import SupConLoss
from . import losses

# Feature fusion using MLP
class feat_learning_layer(nn.Module):

    def __init__(self, input_dim, hidden_size, output_dim):
        #super(Feedforward, self).__init__()
        super().__init__()

        self.input_size = input_dim
        self.hidden_size  = hidden_size
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        #self.relu = nn.LeakyReLU(0.1)
        self.relu = nn.GELU()
        #self.fc2 = nn.Linear(1024, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_dim)

        self.mlp_dropout = nn.Dropout(0.25)
        self.lay_norm_hid = nn.LayerNorm(hidden_size)
        self.lay_norm_ip = nn.LayerNorm(input_dim)
        self.lay_norm_op = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.lay_norm_ip(x)
        hidden = self.fc1(x)
        #hidden = self.droput(hidden)
        relu = self.relu(hidden)
        #relu = self.lay_norm_hid(relu)
        #relu = self.mlp_dropout(relu)
        output = self.fc2(relu)
        #output = self.lay_norm_op(output)
        #relu_1 = self.relu(output_1)
        #output = self.fc3(relu_1)

        return output

class GraphLearningLayer(nn.Module):
    def __init__(self, in_dim: int, learning_dim: int, gamma: float, eta: float, dropout: float = 0.1):
        super().__init__()

        learning_dim = 256
        graph_in_dim = 256
        
        self.projection = nn.Linear(graph_in_dim, learning_dim, bias=False)
        self.projection_text = nn.Linear(256, 128, bias=False)   #Change
        self.projection_vis = nn.Linear(256, 128, bias=False)
        self.projection_post = nn.Linear(3, 256, bias=False)
        self.projection_tot = nn.Linear(512+256, 256, bias=False)
        self.prog_256 = nn.Linear(256,256, bias=False)
        
        self.learn_w = nn.Parameter(torch.empty(learning_dim))
        self.gamma = gamma
        self.eta = eta
        self.inint_parameters()

        
        # Positional Embedding
        max_len = 200
        # Compute the positional encodings once in log space.
        
        pos_emd_dim = graph_in_dim
        self.pos_projection = nn.Linear(graph_in_dim, learning_dim, bias=False)

        position_embedding_1 = torch.zeros(max_len, pos_emd_dim)
        position_1 = torch.arange(0, max_len).unsqueeze(1).float()
        div_term_1 = torch.exp(torch.arange(0, pos_emd_dim, 2).float() *
                             -(math.log(10000.0) / pos_emd_dim))
        position_embedding_1[:, 0::2] = torch.sin(position_1 * div_term_1)
        position_embedding_1[:, 1::2] = torch.cos(position_1 * div_term_1)

        #position_embedding_1 = pos_projection(position_embedding_1)     
        position_embedding_1 = position_embedding_1.unsqueeze(0)  # 1, max_len, pos_emd_dim
        
        self.register_buffer('position_embedding_1', position_embedding_1)
        #self.position_embedding_1 = position_embedding_1

        self.pe_droput = nn.Dropout(0.25)
        
        self.lay_norm = nn.LayerNorm(256)
        
        # MLP
        hidden_size = 1024
        output_dim = learning_dim
        self.mlp = feat_learning_layer(graph_in_dim, hidden_size, output_dim)
        self.softmax = nn.Softmax(dim=2)
        

    def inint_parameters(self):
        nn.init.uniform_(self.learn_w, a=0, b=1)

    def forward(self, x: Tensor, adj: Tensor, box_labels: Tensor, box_num: Tensor = None):
        '''

        :param x: nodes set, (B*N, D)
        :param adj: init adj, (B, N, N, default is 1)
        :param box_num: (B, 1)
        :return:
                out, soft adj matrix
                gl loss
        '''
        B, N, D = x.shape    
        #print(box_labels.shape, x.shape, box_num)
        
        # (B, N, D)
        
        # Feature Fusion using linear projection
        #x_text = self.projection_text(self.norm(x[:,:,:256]))
        #x_vis = self.projection_vis(self.norm(x[:,:,256:512]))
        #x_post = self.projection_post(x[:,:,512:])
        #x_hat = torch.cat((x_text, x_vis, x_post), 2)
        #x_hat = torch.cat((x_text + x_post, x_vis + x_post), 2)

        # No initial embedding for text and visual features
        #x_hat = self.projection_tot(torch.cat((x[:,:,:512],x_post), 2))
        #x_hat = self.projection(x)
        #x_hat = x

        
        #x = torch.cat((x[:,:,:512], self.projection_post(x[:,:,512:])), 2)

        # Layer normalization
        #x = self.norm(x)

        # L2 normalizing the features
        x = F.normalize(x, p=2, dim=2)

        #Feature Fusion using MLP
        x_hat_1 = self.mlp(x)

        # Like skip connection
        x_hat = x_hat_1 + x

        #x_hat = self.norm(x_hat)

        # L2 normalizing the features
        #x_hat = F.normalize(x_hat, p=2, dim=2)

        #x_hat = self.lay_norm(x)
        #print('before', torch.isnan(x_hat))

        # Adding Positional Embedding
        #x_hat = self.pe_droput(x_hat + self.position_embedding_1[:, :x_hat.size(1), :])
        #pos_emd = self.pos_projection(self.position_embedding_1)
        #pos_emd = pos_emd.expand(x_proj.size(0), -1, -1)
        #print(pos_emd[0,0:10,:])
        
        #x_hat = self.pe_droput(torch.cat((x_proj, pos_emd[:, :x_proj.size(1), :]), 2))
        #x_hat = self.pe_droput(x_proj + pos_emd[:, :x_proj.size(1), :])
        #print(x_hat.shape)
        
        _, _, learning_dim = x_hat.shape

        distance_dot_product = torch.bmm(x_hat, torch.transpose(x_hat, 1, 2))
        #distance_dot_product_soft_max = self.softmax(distance_dot_product)

        #print(x_hat.shape, distance_dot_product.shape)
        #print(torch.dot(x_hat[1,15,:], x_hat[1,20,:]), 'dot_prod', distance_dot_product[1,15,20], distance_dot_product[1,20,15])
        #print(torch.dot(x_hat[1,5,:], x_hat[1,10,:]), 'dot_prod', distance_dot_product[1,5,10], distance_dot_product[1,10,5])
        #print(distance_dot_product_soft_max[1,5,:], torch.sum(distance_dot_product_soft_max[1,5,:]))

        '''
        # (B, N, N, learning_dim)
        x_i = x_hat.unsqueeze(2).expand(B, N, N, learning_dim)
        x_j = x_hat.unsqueeze(1).expand(B, N, N, learning_dim)
        # (B, N, N, learning_dim)
        distance = torch.abs(x_i - x_j)
        #print(x_i[0,2,3,:], '\n', x_j[0,3,2,:])

        # Computing dot product between x_i and x_j 
        #distance_1 = torch.inner(x_i, x_j)
        
        #distance_1 = torch.zeros(B, N, N, device=x.device)
        #for k in range(B):
        #    for i in range(N):
        #        for j in range(N):
        #            distance_1[k,i,j] = torch.dot(x[k,i,:], x[k,j,:])
        

        #print(distance_1.shape)

        #distance = torch.cat((x_i, x_j), 3)
        #learning_dim = distance_concat.shape(3)

        # add -1 flag to distance, if node is not exist. to separate normal node distances from not exist node distance.
        if box_num is not None:
            # mask = self.compute_static_mask(box_num)
            mask = self.compute_dynamic_mask(box_num)
            #print('mask shape', mask.shape, distance.shape, distance_1.shape)
            distance = distance + mask

        
        # (B, N, N)
        distance = torch.einsum('bijd, d->bij', distance, self.learn_w)
        '''

        #out = F.leaky_relu(distance_dot_product)
        out = distance_dot_product
        #print('--', distance.shape, out.shape)

        # for numerical stability, due to softmax operation mable produce large value
        max_out_v, _ = out.max(dim=-1, keepdim=True)
        out = out - max_out_v.detach()

        soft_adj = torch.exp(out)
        soft_adj = adj * soft_adj

        sum_out = soft_adj.sum(dim=-1, keepdim=True)
        soft_adj = soft_adj / sum_out + 1e-10
        #print('soft_adj_mat', soft_adj[0,1,:])

        
        gl_loss = None
        if self.training:
            gl_loss = self._graph_learning_loss(x_hat, soft_adj, box_num, box_labels)
            #print(gl_loss)

        #print('weights', self.learn_w)
        # Multiplying learned w (learn_w) with the feature vectors
        #weights = self.learn_w.expand_as(x_hat)

        #weights = self.learn_w/torch.sum(self.learn_w)
        #weights = weights.expand_as(x_hat)
        #print('---', weights[0,0,0:10])
        #x_hat = torch.mul(weights, x_hat)
        #x_hat = self.norm(x_hat)

        return soft_adj, gl_loss, x_hat_1

    @staticmethod
    def compute_static_mask(box_num: Tensor):
        '''
        compute -1 mask, if node(box) is not exist, the length of mask is documents.MAX_BOXES_NUM,
        this will help with one nodes multi gpus training mechanism, and ensure batch shape is same. but this operation
        lead to waste memory.
        :param box_num: (B, 1)
        :return: (B, N, N, 1)
        '''
        max_len = documents.MAX_BOXES_NUM

        # (B, N)
        mask = torch.arange(0, max_len, device=box_num.device).expand((box_num.shape[0], max_len))

        # (B, N)
        box_num = box_num.expand_as(mask)
        mask = mask < box_num

        # (B, 1, N)
        row_mask = mask.unsqueeze(1)

        # (B, N, 1)
        column_mask = mask.unsqueeze(2)

        # (B, N, N)
        mask = (row_mask & column_mask)

        # -1 if not exist node, or 0
        mask = ~mask * -1

        return mask.unsqueeze(-1)

    @staticmethod
    def compute_dynamic_mask(box_num: Tensor):
        '''
        compute -1 mask, if node(box) is not exist, the length of mask is calculate by max(box_num),
        this will help with multi nodes multi gpus training mechanism, ensure batch of different gpus have same shape.
        :param box_num: (B, 1)
        :return: (B, N, N, 1)
        '''
        max_len = torch.max(box_num)

        # (B, N)
        mask = torch.arange(0, max_len, device=box_num.device).expand((box_num.shape[0], max_len))

        # (B, N)
        box_num = box_num.expand_as(mask)
        mask = mask < box_num

        # (B, 1, N)
        row_mask = mask.unsqueeze(1)

        # (B, N, 1)
        column_mask = mask.unsqueeze(2)

        # (B, N, N)
        mask = (row_mask & column_mask)

        # -1 if not exist node, or 0
        mask = ~mask * -1

        return mask.unsqueeze(-1)

    def _graph_learning_loss(self, x_hat: Tensor, adj: Tensor, box_num: Tensor, box_labels: Tensor):
        '''
        calculate graph learning loss
        :param x_hat: (B, N, D)
        :param adj: (B, N, N)
        :param box_num: (B, 1)
        :return:
            gl_loss
        '''

        B, N, D = x_hat.shape
        '''
        # (B, N, N, out_dim)
        x_i = x_hat.unsqueeze(2).expand(B, N, N, D)
        x_j = x_hat.unsqueeze(1).expand(B, N, N, D)

        # (B, 1)
        box_num_div = 1 / torch.pow(box_num.float(), 2)

        # (B, N, N)
        dist_loss = adj + self.eta * torch.norm(x_i - x_j, dim=3) # remove square operation duo to it can cause nan loss.
        dist_loss = torch.exp(dist_loss)
        # (B,)
        dist_loss = torch.sum(dist_loss, dim=(1, 2)) * box_num_div.squeeze(-1)
        # (B,)
        f_norm = torch.norm(adj, dim=(1, 2)) # remove square operation duo to it can cause nan loss.

        gl_loss = dist_loss + self.gamma * f_norm
        return gl_loss
        '''

        # define loss with a temperature `temp`
        temp = 0.1
        criterion = losses.SupConLoss(temperature=temp)

        # features: [bsz: batch_size, n_views, f_dim]
        # `n_views` is the number of crops from each image
        # better be L2 normalized in f_dim dimension

        f_norm = torch.norm(adj, dim=(1, 2))
        loss = torch.zeros(2 , device=f_norm.device)

        for i in range(box_num.shape[0]):
            features = x_hat[i, :box_num[i], :]
            # features dim: 1 x num_nodes x d
            features = features.unsqueeze(1)
            labels = box_labels[i,:box_num[i]]
            labels = labels.unsqueeze(1)
            #print(torch.bincount(features))
            #t = torch.isnan(features)
            #print('Number of nan in features', len(t[t == True]))
            
            #print(torch.nonzero(torch.isnan(features)))
            #print(features.shape, labels.shape)

            # SupContrast
            #loss.append(criterion(features, labels))
            loss[i] = criterion(features, labels)
            #print(loss)
            
        '''
        features = x_hat
        labels = box_labels
        print(features.shape, labels.shape)
        print('loss', criterion(features, labels))
        '''
        
        #print('Loss --------', loss, 'f_norm', f_norm)

        return loss + (self.gamma * f_norm)

#For Layer 0
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        '''
        perform graph convolution operation
        :param in_dim:
        :param out_dim:
        '''
        super().__init__()

        ip_dim = 256
        op_dim = 256
        self.w_alpha = nn.Parameter(torch.empty(256, 256))
        self.w_vi = nn.Parameter(torch.empty(ip_dim, op_dim))
        self.w_vj = nn.Parameter(torch.empty(ip_dim, op_dim))
        self.bias_h = nn.Parameter(torch.empty(op_dim))
        self.w_node = nn.Parameter(torch.empty(256, out_dim))

        self.inint_parameters()

    def inint_parameters(self):
        nn.init.kaiming_uniform_(self.w_alpha, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_vi, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_vj, a=math.sqrt(5))
        nn.init.uniform_(self.bias_h, a=0, b=1)
        nn.init.kaiming_uniform_(self.w_node, a=math.sqrt(5))

    def forward(self, x: Tensor, alpha: Tensor, adj: Tensor, box_num: Tensor):
        '''

        :param x: nodes set (node embedding), (B, N, in_dim)
        :param alpha: relation embedding, (B, N, N, in_dim)
        :param adj: learned soft adj matrix, (B, N, N)
        :param box_num: (B, 1)
        :return:
                x_out: updated node embedding, (B, N, out_dim)
                alpha: updated relation embedding, (B, N, N, out_dim)
        '''

        B, N, in_dim = x.shape

        # (B, N, N, in_dim)
        x_i = x.unsqueeze(2).expand(B, N, N, in_dim)
        x_j = x.unsqueeze(1).expand(B, N, N, in_dim)
        #print('in_dim', in_dim, x.shape, x_i.shape, x_j.shape, alpha.shape)
        
        #print('Layer 1', x_i.shape)
        # (B, N, N, in_dim)
        x_i = torch.einsum('bijd, dk->bijk', x_i, self.w_vi)
        x_j = torch.einsum('bijd, dk->bijk', x_j, self.w_vj)

        # update hidden features between nodes, (B, N, N, in_dim）
        #print('Layer 1 ----',  x_i.shape, x_j.shape, alpha.shape)
        H = F.relu(x_i + x_j + alpha + self.bias_h)

        # update node embedding x, （B, N, out_dim）
        AH = torch.einsum('bij, bijd-> bid', adj, H)
        new_x = torch.einsum('bid,dk->bik', AH, self.w_node)
        new_x = F.relu(new_x)

        # update relation embedding, (B, N, N, out_dim)
        new_alpha = torch.einsum('bijd,dk->bijk', H, self.w_alpha)
        new_alpha = F.relu(new_alpha)
        #print('Updated alpha shape', new_alpha.shape)

        return new_x, new_alpha

#For Layer 1
class GCNLayer_layer_1(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        '''
        perform graph convolution operation
        :param in_dim:
        :param out_dim:
        '''
        super().__init__()

        #self.w_alpha = nn.Parameter(torch.empty(256, out_dim))
        ip_dim = 256
        op_dim = 256
        self.w_vi = nn.Parameter(torch.empty(ip_dim, op_dim))
        self.w_vj = nn.Parameter(torch.empty(ip_dim, op_dim))
        self.bias_h = nn.Parameter(torch.empty(256))
        self.w_node = nn.Parameter(torch.empty(256, out_dim))

        self.inint_parameters()

    def inint_parameters(self):
        #nn.init.kaiming_uniform_(self.w_alpha, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_vi, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_vj, a=math.sqrt(5))
        nn.init.uniform_(self.bias_h, a=0, b=1)
        nn.init.kaiming_uniform_(self.w_node, a=math.sqrt(5))

    def forward(self, x: Tensor, alpha: Tensor, adj: Tensor, box_num: Tensor):
        '''

        :param x: nodes set (node embedding), (B, N, in_dim)
        :param alpha: relation embedding, (B, N, N, in_dim)
        :param adj: learned soft adj matrix, (B, N, N)
        :param box_num: (B, 1)
        :return:
                x_out: updated node embedding, (B, N, out_dim)
                alpha: updated relation embedding, (B, N, N, out_dim)
        '''

        B, N, in_dim = x.shape

        # (B, N, N, in_dim)
        x_i = x.unsqueeze(2).expand(B, N, N, in_dim)
        x_j = x.unsqueeze(1).expand(B, N, N, in_dim)
        #print('in_dim', in_dim, x.shape, x_i.shape, x_j.shape, alpha.shape)
        
        #print('Layer 1', x_i.shape, alpha.shape)
        # (B, N, N, in_dim)
        x_i = torch.einsum('bijd, dk->bijk', x_i, self.w_vi)
        x_j = torch.einsum('bijd, dk->bijk', x_j, self.w_vj)

        # update hidden features between nodes, (B, N, N, in_dim）
        #print('Layer 1 ---', x_i.shape, x_j.shape, alpha.shape)
        H = F.relu(x_i + x_j + alpha + self.bias_h)

        # update node embedding x, （B, N, out_dim）
        AH = torch.einsum('bij, bijd-> bid', adj, H)
        new_x = torch.einsum('bid,dk->bik', AH, self.w_node)
        new_x = F.relu(new_x)

        # update relation embedding, (B, N, N, out_dim)
        #new_alpha = torch.einsum('bijd,dk->bijk', H, self.w_alpha)
        #new_alpha = F.relu(new_alpha)

        return new_x, new_x


class GLCN(nn.Module):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 gamma: float = 0.0001,
                 eta: float = 1,
                 learning_dim: int = 128,
                 num_layers=2,
                 dropout: float = 0.1):
        '''
        perform graph learning and multi-time graph convolution operation
        :param in_dim:
        :param out_dim:
        :param gamma:
        :param eta:
        :param learning_dim:
        :param num_layers:
        '''
        super().__init__()

        self.gl_layer = GraphLearningLayer(in_dim=in_dim, gamma=gamma, eta=eta, learning_dim=learning_dim)

        modules = []
        in_dim_cur = in_dim
        
        #Feat learning variables
        #input_dim = 768
        #hidden_size = 512
        output_dim = in_dim
        
        '''
        for i in range(num_layers+1):
            if i == 0:
                m = feat_learning_layer(input_dim, hidden_size, output_dim)
            else:
                m = GCNLayer(in_dim_cur, out_dim)
                in_dim_cur = out_dim
                out_dim = in_dim_cur
                
            modules.append(m)
        self.gcn = nn.ModuleList(modules)
        '''

        '''
        for i in range(num_layers):
            m = GCNLayer(in_dim_cur, out_dim)
            in_dim_cur = out_dim
            out_dim = in_dim_cur
            modules.append(m)
        '''
        for i in range(num_layers):
            if i==0:
                m = GCNLayer(in_dim_cur, 256)
                in_dim_cur = 256
                out_dim = in_dim_cur
            elif i==1:
                m = GCNLayer_layer_1(in_dim_cur, 256)
                in_dim_cur = 256
                out_dim = in_dim_cur
            modules.append(m)
        
        self.gcn = nn.ModuleList(modules)
        

        self.alpha_transform = nn.Linear(8, 256, bias=False)
        self.pos_transform = nn.Linear(3, 256, bias=False)
        self.vis_transform = nn.Linear(256, 256, bias=False)
        self.text_transform = nn.Linear(256, 256, bias=False)
        self.proj_1 = nn.Linear(256, 256, bias=False)
        self.proj_2 = nn.Linear(256, 256, bias=False)

        self.input_transform = nn.Linear(384, 256, bias=False)

        self.ot_dropout = nn.Dropout(0.1)

        #self.mlp_ctc_loss = mlp_ctc_loss(input_dim, hidden_size, output_dim)

    def forward(self, x: Tensor, rel_features: Tensor, adj: Tensor, box_num: Tensor, box_labels: Tensor, **kwargs):
        '''

        :param x: nodes embedding, (B*N, D)
        :param rel_features: relation embedding, (B, N, N, 6)
        :param adj: default adjacent matrix, (B, N, N)
        :param box_num: (B, 1)
        :param kwargs:
        :return:
        '''
        
        #print(rel_features[0,1,:,5])
        # relation features embedding, (B, N, N, in_dim)
        alpha = self.alpha_transform(rel_features)  #dim: [2, 40, 40, 256]
        #print(alpha.shape)
        #print('x shape', x.shape)
        pos = x[:,:,512:]
        #x = x[:,:,:512]
        #print('txt and vis features', x[0,0,:])
        
        #print('Pos shape', pos.shape)

        #Project the position and height features
        pos = self.pos_transform(pos)
        #print('Pos shape', pos.shape)

        # Projecting text, visual features then concatenating
        #x_1 = torch.cat((self.text_transform(x[:,:,:256]), self.vis_transform(x[:,:,256:512]), pos), 2)
        #print('x new shape', x.shape)

        # Projecting then doing the summation
        x_1 = self.text_transform(x[:,:,:256]) + self.vis_transform(x[:,:,256:512]) + pos
        #print('feature fusion sum', torch.sum(x_1))

        # Adding Dropout
        #x_1 = self.ot_dropout(x_1)
        #x_1 = F.dropout(x_1, p=0.2, training=self.training)


        #Project the input data
        #x = self.input_transform(x_org)

        soft_adj, gl_loss, x_mod = self.gl_layer(x_1, adj, box_labels, box_num)
        adj = adj * soft_adj
        #x = x_1 

        #print('sum of feat from GLCN', torch.sum(x_mod))
        # Combining the feat from GLCN with the visual feat
        #x = self.proj_1(x_1) + self.proj_2(x_mod)
        x = x_1 + x_mod

        #x = self.input_transform(x)

        '''
        for i, gcn_layer in enumerate(self.gcn):
            if i == 0:
                x = gcn_layer(x)
                soft_adj, gl_loss = self.gl_layer(x, adj, box_num)
                adj = adj * soft_adj
                #print('i = ', i, 'x shape ', x.shape)
            else: 
                x, alpha = gcn_layer(x, alpha, adj, box_num)
                #print('i = ', i, 'x shape After ', x.shape)
        '''
        
        for i, gcn_layer in enumerate(self.gcn):
            #print('i', i)
            x, alpha = gcn_layer(x, alpha, adj, box_num)
        
        #print('sum of feat from GCN', torch.sum(x))
        '''
        #Classifier with CTC Loss
        model = self.mlp_ctc_loss(x)
        ctc_loss = nn.CTCLoss()
        loss = ctc_loss(input, target, input_lengths, target_lengths)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
        '''

        return x, soft_adj, gl_loss
