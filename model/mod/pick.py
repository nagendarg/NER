# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/8/2020 10:54 PM

from typing import *

import torch
import torch.nn as nn
import numpy as np

from .encoder import Encoder
#from .encoder_pos import Encoder
from .graph import GLCN
#from .graph_pos import GLCN
from .decoder import Decoder
from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls


class PICKModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        embedding_kwargs = kwargs['embedding_kwargs']
        encoder_kwargs = kwargs['encoder_kwargs']
        graph_kwargs = kwargs['graph_kwargs']
        decoder_kwargs = kwargs['decoder_kwargs']
        self.make_model(embedding_kwargs, encoder_kwargs, graph_kwargs, decoder_kwargs)

    def make_model(self, embedding_kwargs, encoder_kwargs, graph_kwargs, decoder_kwargs):
        # Given the params of each component, creates components.
        # embedding_kwargs-> word_emb
        embedding_kwargs['num_embeddings'] = len(keys_vocab_cls)
        self.word_emb = nn.Embedding(**embedding_kwargs)

        encoder_kwargs['char_embedding_dim'] = embedding_kwargs['embedding_dim']
        self.encoder = Encoder(**encoder_kwargs)

        graph_kwargs['in_dim'] = encoder_kwargs['out_dim']
        graph_kwargs['out_dim'] = encoder_kwargs['out_dim']
        self.graph = GLCN(**graph_kwargs)

        decoder_kwargs['bilstm_kwargs']['input_size'] = encoder_kwargs['out_dim']
        if decoder_kwargs['bilstm_kwargs']['bidirectional']:
            decoder_kwargs['mlp_kwargs']['in_dim'] = decoder_kwargs['bilstm_kwargs']['hidden_size'] * 2
        else:
            decoder_kwargs['mlp_kwargs']['in_dim'] = decoder_kwargs['bilstm_kwargs']['hidden_size']
        decoder_kwargs['mlp_kwargs']['out_dim'] = len(iob_labels_vocab_cls)
        decoder_kwargs['crf_kwargs']['num_tags'] = len(iob_labels_vocab_cls)
        self.decoder = Decoder(**decoder_kwargs)

    def _aggregate_avg_pooling(self, input, text_mask):
        '''
        Apply mean pooling over time (text length), (B*N, T, D) -> (B*N, D)
        :param input: (B*N, T, D)
        :param text_mask: (B*N, T)
        :return: (B*N, D)
        '''
        # filter out padding value, (B*N, T, D)
        input = input * text_mask.detach().unsqueeze(2).float()
        # (B*N, D)
        sum_out = torch.sum(input, dim=1)
        # (B*N, )
        text_len = text_mask.float().sum(dim=1)
        # (B*N, D)
        text_len = text_len.unsqueeze(1).expand_as(sum_out)
        text_len = text_len + text_len.eq(0).float()  # avoid divide zero denominator
        # (B*N, D)
        mean_out = sum_out.div(text_len)
        return mean_out

    @staticmethod
    def compute_mask(mask: torch.Tensor):
        '''
        :param mask: (B, N, T)
        :return: True for masked key position according to pytorch official implementation of Transformer
        '''
        B, N, T = mask.shape
        mask = mask.reshape(B * N, T)
        mask_sum = mask.sum(dim=-1)  # (B*N,)

        # (B*N,)
        graph_node_mask = mask_sum != 0
        # (B * N, T)
        graph_node_mask = graph_node_mask.unsqueeze(-1).expand(B * N, T)  # True for valid node
        # If src key are all be masked (indicting text segments is null), atten_weight will be nan after softmax
        # in self-attention layer of Transformer.
        # So we do not mask all padded sample. Instead we mask it after Transformer encoding.
        src_key_padding_mask = torch.logical_not(mask.bool()) & graph_node_mask  # True for padding mask position
        return src_key_padding_mask, graph_node_mask

    def forward(self, **kwargs):
        # input
        whole_image = kwargs['whole_image']  # (B, 3, H, W)
        relation_features = kwargs['relation_features']  # initial relation embedding (B, N, N, 6)
        text_segments = kwargs['text_segments']  # text segments (B, N, T)
        text_length = kwargs['text_length']  # (B, N)
        iob_tags_label = kwargs['iob_tags_label'] if self.training else None  # (B, N, T)
        mask = kwargs['mask']  # (B, N, T)
        boxes_coordinate = kwargs['boxes_coordinate']  # (B, num_boxes, 8)
        boxes_coordinate_org = kwargs['boxes_coordinate_org']

        #print(boxes_coordinate_org.shape)
        ##### Forward Begin #####
        ### Encoder module ###
        # word embedding
        #print('Text Segment Shape: ', text_segments.shape)
        #Text Segment Shape:  torch.Size([2, 40, 41])
        #Text Segment Shape:  torch.Size([2, 53, 42])

        text_emb = self.word_emb(text_segments)
        #print('Text Segment After Embedding: ', text_emb.shape)
        #Text Segment After Embedding:  torch.Size([2, 40, 41, 512])


        # src_key_padding_mask is text padding mask, True is padding value (B*N, T)
        # graph_node_mask is mask for graph, True is valid node, (B*N, T)
        src_key_padding_mask, graph_node_mask = self.compute_mask(mask)

        #Modification By Nag
        # text_mask, True for valid, (including all not valid node), (B*N, T)
        text_mask = torch.logical_not(src_key_padding_mask).byte()
        #text_emb_new = self._aggregate_avg_pooling(text_emb, text_mask)
        #print('New Text Embedding Shape: ', text_emb_new.shape)

        # set of nodes, (B*N, T, D)
        #print(boxes_coordinate[0,:,:])
        #print(boxes_coordinate[1,:,:])
        #print(box)

        #x, x_concat = self.encoder(images=whole_image, boxes_coordinate=boxes_coordinate, boxes_coordinate_org=boxes_coordinate_org, transcripts=text_emb,
        #                 src_key_padding_mask=src_key_padding_mask, text_mask=text_mask)

        x, x_concat = self.encoder(images=whole_image, boxes_coordinate=boxes_coordinate, transcripts=text_emb,
                         src_key_padding_mask=src_key_padding_mask, text_mask=text_mask)
        #x is [B*N, T, D]
        #print(x.shape, '\n ------------- Encoder Done -----------')
        #torch.Size([80, 41, 512])

        ### Graph module ###
        # text_mask, True for valid, (including all not valid node), (B*N, T)
        text_mask = torch.logical_not(src_key_padding_mask).byte()
        # (B*N, T, D) -> (B*N, D)
        #x_gcn = self._aggregate_avg_pooling(x, text_mask)
        x_gcn = x_concat
        #print('x_gcn shape: ', x_gcn.shape)
        #x_gcn shape:  torch.Size([80, 512])

        # (B*N, 1)，True is valid node
        graph_node_mask = graph_node_mask.any(dim=-1, keepdim=True)
        # (B*N, D), filter out not valid node
        x_gcn = x_gcn * graph_node_mask.byte()
        #print('x_gcn shape after filterout: ', x_gcn.shape)
        #x_gcn shape after filterout:  torch.Size([80, 512])

        # initial adjacent matrix (B, N, N)
        B, N, T = mask.shape
        init_adj = torch.ones((B, N, N), device=text_emb.device)
        boxes_num = mask[:, :, 0].sum(dim=1, keepdim=True)  # (B, 1)
        # (B, N, D)
        x_gcn = x_gcn.reshape(B, N, -1)
        #print(x_gcn.shape, relation_features.shape, boxes_num.shape)
        #[2, 40, 512]   [2, 40, 40, 6]  [2, 1]
        #print(relation_features[0,1,1,:], boxes_num)
        #[0.0000, 0.0000, 0.1041, 0.9984, 0.1068, 0.9993],  [[40],[37]] boxes_num: No of BBs in documents

        # (B, N, D), (B, N, N), (B,)
        x_gcn, soft_adj, gl_loss = self.graph(x_gcn, relation_features, init_adj, boxes_num)
        adj = soft_adj * init_adj
        #print('x shape after GCN ', x_gcn.shape, soft_adj.shape, adj.shape)
        #[2, 40, 512]   [2, 40, 40] [2, 40, 40]

        #print(x.shape, x.reshape(B, N, T, -1).shape, text_length.shape)
        #[80, 41, 512]  [2, 40, 41, 512]    [2, 40]
        #text_length is contains the length of each BB (40 is the max BBs in doc_1, doc_2)
        #print(x[0,39,:], iob_tags_label.shape)
        #mask gives there is a paading in each text segment (BB), [2, 40, 41]
        #iob_tags_label: [2, 40, 41]

        ### Decoder module ###
        logits, new_mask, log_likelihood = self.decoder(x.reshape(B, N, T, -1), x_gcn, mask, text_length,
                                                        iob_tags_label)
        #print(new_mask.shape)
        #torch.Size([2, 497])

        ##### Forward End #####

        output = {"logits": logits, "new_mask": new_mask, "adj": adj}

        if self.training:
            output['gl_loss'] = gl_loss
            crf_loss = -log_likelihood
            output['crf_loss'] = crf_loss
        return output

    def __str__(self):
        '''
        Model prints with number of trainable parameters
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
