# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/7/2020 5:54 PM

from typing import *
import math
import umap
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.ops import roi_pool

from . import resnet


class Encoder(nn.Module):

    def __init__(self,
                 char_embedding_dim: int,
                 out_dim: int,
                 image_feature_dim: int = 512,
                 nheaders: int = 8,
                 nlayers: int = 6,
                 feedforward_dim: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 100,
                 image_encoder: str = 'resnet50',
                 roi_pooling_mode: str = 'roi_align',
                 roi_pooling_size: Tuple[int, int] = (7, 7)):
        '''
        convert image segments and text segments to node embedding.
        :param char_embedding_dim:
        :param out_dim:
        :param image_feature_dim:
        :param nheaders:
        :param nlayers:
        :param feedforward_dim:
        :param dropout:
        :param max_len:
        :param image_encoder:
        :param roi_pooling_mode:
        :param roi_pooling_size:
        '''
        super().__init__()

        self.dropout = dropout
        assert roi_pooling_mode in ['roi_align', 'roi_pool'], 'roi pooling model: {} not support.'.format(
            roi_pooling_mode)
        self.roi_pooling_mode = roi_pooling_mode
        assert roi_pooling_size and len(roi_pooling_size) == 2, 'roi_pooling_size not be set properly.'
        self.roi_pooling_size = tuple(roi_pooling_size)  # (h, w)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=char_embedding_dim,
                                                               nhead=nheaders,
                                                               dim_feedforward=feedforward_dim,
                                                               dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=nlayers)

        if image_encoder == 'resnet18':
            self.cnn = resnet.resnet18(output_channels=out_dim)
        elif image_encoder == 'resnet34':
            self.cnn = resnet.resnet34(output_channels=out_dim)
        elif image_encoder == 'resnet50':
            self.cnn = resnet.resnet50(output_channels=out_dim)
        elif image_encoder == 'resnet101':
            self.cnn = resnet.resnet101(output_channels=out_dim)
        elif image_encoder == 'resnet152':
            self.cnn = resnet.resnet152(output_channels=out_dim)
        else:
            raise NotImplementedError()

        self.conv = nn.Conv2d(image_feature_dim, out_dim, self.roi_pooling_size)
        self.bn = nn.BatchNorm2d(out_dim)

        self.projection = nn.Linear(2 * out_dim, out_dim)
        self.norm = nn.LayerNorm(512)
        self.norm_1 = nn.LayerNorm(1024)

        # Compute the positional encodings once in log space.
        position_embedding = torch.zeros(max_len, char_embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, char_embedding_dim, 2).float() *
                             -(math.log(10000.0) / char_embedding_dim))
        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_embedding.unsqueeze(0).unsqueeze(0)  # 1, 1, max_len, char_embedding_dim
        self.register_buffer('position_embedding', position_embedding)

        self.pe_droput = nn.Dropout(self.dropout)

    def aggregate_avg_pooling(input, text_mask):
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

    def forward(self, images: torch.Tensor, boxes_coordinate: torch.Tensor, transcripts: torch.Tensor,
                src_key_padding_mask: torch.Tensor, text_mask: torch.Tensor):
        '''

        :param images: whole_images, shape is (B, N, H, W, C), where B is batch size, N is the number of segments of
                the documents, H is height of image, W is width of image, C is channel of images (default is 3).
        :param boxes_coordinate: boxes coordinate, shape is (B, N, 8), where 8 is coordinates (x1, y1, x2, y2, x3, y3, x4, y4)
        :param transcripts: text segments, shape is (B, N, T, D), where T is the max length of transcripts,
                                D is dimension of model
        :param src_key_padding_mask: text padding mask, shape is (B*N, T), True for padding value.
            if provided, specified padding elements in the key will be ignored by the attention.
            This is an binary mask. When the value is True, the corresponding value on the attention layer of Transformer
            will be filled with -inf.
        need_weights: output attn_output_weights.
        :return: set of nodes X, shape is (B*N, T, D)
        '''

        B, N, T, D = transcripts.shape
        #print(transcripts[0,20,38,:])
        #print('Transcript dimensions B, N, T, D', B, N, T, D)
        #Transcript dimensions B, N, T, D: 2 40 41 512
        #Transcript dimensions B, N, T, D 2 59 39 512

        # get image embedding using cnn
        # (B, 3, H, W)
        _, _, origin_H, origin_W = images.shape
        #print('Original Image Shape', origin_H, origin_W)
        #Original Image Shape 960 480

        # (B, C, H/16, W/16)
        images = self.cnn(images)
        _, C, H, W = images.shape
        #print('Image Featuremap shape- ResNet- H, W, C', H, W, C)
        #Image Featuremap shape- ResNet- H, W, C: 60 30 512

        # generate rois for roi pooling, rois shape is (B, N, 5), 5 means (batch_index, x0, y0, x1, y1)
        rois_batch = torch.zeros(B, N, 5, device=images.device)
        loc_feat = torch.zeros(B*N, 2, device=images.device)
        for i in range(B):  # (B, N, 8)
            # (N, 8)
            doc_boxes = boxes_coordinate[i]
            # (N, 4)
            pos = torch.stack([doc_boxes[:, 0], doc_boxes[:, 1], doc_boxes[:, 4], doc_boxes[:, 5]], dim=1)
            rois_batch[i, :, 1:5] = pos
            rois_batch[i, :, 0] = i

        spatial_scale = float(H / origin_H)
        # use roi pooling get image segments
        # (B*N, C, roi_pooling_size, roi_pooling_size)
        if self.roi_pooling_mode == 'roi_align':
            image_segments = roi_align(images, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)
        else:
            image_segments = roi_pool(images, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)

        # (B*N, D, 1, 1)
        image_segments = F.relu(self.bn(self.conv(image_segments)))
        # # (B*N, D,)
        image_segments = image_segments.squeeze()
        image_segments_new = image_segments
        #print('Image segment shape', image_segments.shape)
        #Image segment shape torch.Size([80, 512])

        # (B*N, 1, D)
        image_segments = image_segments.unsqueeze(dim=1)
        #Image Segment Shape:  torch.Size([80, 1, 512])

        #print('Transcript Shape: ', transcripts.shape)
        #Transcript Shape:  torch.Size([2, 40, 41, 512])

        # add positional embedding
        transcripts_segments = self.pe_droput(transcripts + self.position_embedding[:, :, :transcripts.size(2), :])
        #print('Transcript_Segments Shape: ', transcripts_segments.shape)
        #Transcript_Segments Shape:  torch.Size([2, 40, 41, 512])

        # (B*N, T ,D)
        
        transcripts_segments = transcripts_segments.reshape(B * N, T, D)
        
        transcripts_segments_new = transcripts_segments.transpose(0, 1).contiguous()   # (T, B*N, D)
        transcripts_segments_new = self.transformer_encoder(transcripts_segments_new, src_key_padding_mask=src_key_padding_mask) 
        transcripts_segments_new = transcripts_segments_new.transpose(0, 1).contiguous()   # (B*N, T, D)
        
        ############Averaging transcript segments (B * N, T, D) -> (B * N, D) #########
        input = transcripts_segments_new * text_mask.detach().unsqueeze(2).float()
        # (B*N, D)
        sum_out = torch.sum(input, dim=1)
        # (B*N, )
        text_len = text_mask.float().sum(dim=1)
        # (B*N, D)
        text_len = text_len.unsqueeze(1).expand_as(sum_out)
        text_len = text_len + text_len.eq(0).float()  # avoid divide zero denominator
        # (B*N, D)
        transcripts_segments_new = sum_out.div(text_len)
        ######################
        #print('New Transcript Segments Shape: ', transcripts_segments_new.shape, 'New Image Segment Shape: ', image_segments_new.shape)
        #New Transcript Segments Shape:  torch.Size([140, 512]) New Image Segment Shape:  torch.Size([140, 512])

        transcripts_segments_new = self.norm(transcripts_segments_new)
        image_segments_new = self.norm(image_segments_new)
        

        trans_img = torch.cat((transcripts_segments_new, image_segments_new), 1)
        trans_img = self.norm_1(trans_img)
        #print('Concatenated Shape: ', trans_img.shape)
        #trans_img = trans_img.unsqueeze(dim=1)
        #print('Concatenated Shape: ', trans_img.shape)
        #trans_img = self.transformer_encoder(trans_img, src_key_padding_mask=src_key_padding_mask)
        #print('Concatenated Shape after Transformer: ', trans_img.shape) 
        
        #trans_img = self.norm_1(trans_img)
        '''
        tmp = trans_img.cpu().detach().numpy()
        #print('Number of zeros :', np.count_nonzero(tmp))
        tmp = umap.UMAP(n_neighbors=12, min_dist=0.05, metric='correlation',  n_components=512, init='random').fit_transform(tmp)
         
        trans_img = torch.from_numpy(tmp).cuda(0)
        '''
        #print(trans_img.shape, '\n', trans_img[0,:])
        #Normalization
        
        trans_img = F.dropout(trans_img, p=self.dropout)

        #return trans_img
        

        #transcripts_segments_new = aggregate_avg_pooling(transcripts_segments, text_mask)
        
        #New Transcript Segments Shape:  torch.Size([80, 512]) New Image Segment Shape:  torch.Size([80, 512])  [B*N, D]

        # (B*N, T, D)
        #print('Image Segment Shape: ', image_segments.shape)
        #Image Segment Shape:  torch.Size([80, 1, 512])

        image_segments = image_segments.expand_as(transcripts_segments)    
        #print('Image Segment Shape: ', image_segments.shape)
        #Image Segment Shape:  torch.Size([80, 41, 512])
        #Here, the vector [1, 512] is copied 41 times, which gives [80, 41, 512] from [80, 1, 512]

        # here we first add image embedding and text embedding together,
        # then as the input of transformer to get a non-local fusion features, different from paper process.
        out = image_segments + transcripts_segments
        #print('Out Shape: ', out.shape)
        #Out Shape:  torch.Size([80, 41, 512])

        # (T, B*N, D)
        out = out.transpose(0, 1).contiguous()

        # (T, B*N, D)
        out = self.transformer_encoder(out, src_key_padding_mask=src_key_padding_mask)
        #src_key_padding_mask = [80,41]

        # (B*N, T, D)
        out = out.transpose(0, 1).contiguous()
        #print('Out Shape After Transformer: ', out.shape)
        out = self.norm(out)
        out = F.dropout(out, p=self.dropout)

        return out, trans_img
        #return out
