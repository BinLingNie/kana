
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
from .gatlayer import GAT
import numpy as np

class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..."%(data.word_feature_extractor))
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.strategy = 'n'
        # self.batch_size = data.HP_batch_size
        # self.hidden_dim = data.HP_hidden_dim
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim
        self.feature_num = data.feature_num

        self.gaz_emb_dim = data.gaz_emb_dim
        self.gaz_embeddings = nn.Embedding(data.gaz_size, self.gaz_emb_dim)
        if data.pretrain_gaz_embedding is not None:
            self.gaz_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))
        else:
            self.gaz_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.gaz_size, self.gaz_emb_dim)))

        if self.use_char:
            self.input_size += data.HP_char_hidden_dim
            if data.char_feature_extractor == "ALL":
                self.input_size += data.HP_char_hidden_dim
        for idx in range(self.feature_num):
            self.input_size += data.feature_emb_dims[idx]


        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.hidden_dim = self.gaz_emb_dim
        if self.bilstm_flag:
            lstm_hidden = self.hidden_dim // 2
        else:
            lstm_hidden = self.hidden_dim

        self.word_feature_extractor = data.word_feature_extractor
        if self.word_feature_extractor == "GRU":
            self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "CNN":
            # cnn_hidden = data.HP_hidden_dim
            self.word2cnn = nn.Linear(self.input_size, self.hidden_dim)
            self.cnn_layer = data.HP_cnn_layer
            print("CNN layer: ", self.cnn_layer)
            self.cnn_list = nn.ModuleList()
            self.cnn_drop_list = nn.ModuleList()
            self.cnn_batchnorm_list = nn.ModuleList()
            kernel = 3
            pad_size = int((kernel-1)/2)
            for idx in range(self.cnn_layer):
                self.cnn_list.append(nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel, padding=pad_size))
                self.cnn_drop_list.append(nn.Dropout(data.HP_dropout))
                self.cnn_batchnorm_list.append(nn.BatchNorm1d(self.hidden_dim))
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.hidden_dim, data.label_alphabet_size)

        self.gat_1 = GAT(self.hidden_dim, 30, data.label_alphabet_size, 0, 0.1, 5, 2)
        self.gat_2 = GAT(self.hidden_dim, 30, data.label_alphabet_size, 0, 0.1, 5, 2)
        self.gat_3 = GAT(self.hidden_dim, 30, data.label_alphabet_size, 0, 0.1, 5, 2)
        if self.strategy == "v":
            self.weight1 = nn.Parameter(torch.ones(data.label_alphabet_size))
            self.weight2 = nn.Parameter(torch.ones(data.label_alphabet_size))
            self.weight3 = nn.Parameter(torch.ones(data.label_alphabet_size))
            self.weight4 = nn.Parameter(torch.ones(data.label_alphabet_size))
        elif self.strategy == "n":
            self.weight1 = nn.Parameter(torch.ones(1))
            self.weight2 = nn.Parameter(torch.ones(1))
            self.weight3 = nn.Parameter(torch.ones(1))
            self.weight4 = nn.Parameter(torch.ones(1))
        else:
            self.weight = nn.Linear(data.label_alphabet_size * 4, data.label_alphabet_size)
        self.gaz_dropout = nn.Dropout(0.4)

        if self.gpu:
            self.gaz_embeddings = self.gaz_embeddings.cuda()
            self.droplstm = self.droplstm.cuda()
            self.gaz_dropout = self.gaz_dropout.cuda()
            self.gat_1 = self.gat_1.cuda()
            self.gat_2 = self.gat_2.cuda()
            self.gat_3 = self.gat_3.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            if self.word_feature_extractor == "CNN":
                self.word2cnn = self.word2cnn.cuda()
                for idx in range(self.cnn_layer):
                    self.cnn_list[idx] = self.cnn_list[idx].cuda()
                    self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                    self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()
            else:
                self.lstm = self.lstm.cuda()
            if self.strategy in ["v", "n"]:
                self.weight1.data = self.weight1.data.cuda()
                self.weight2.data = self.weight2.data.cuda()
                self.weight3.data = self.weight3.data.cuda()
                self.weight4.data = self.weight4.data.cuda()
            else:
                self.weight = self.weight.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, gaz_list, t_graph, c_graph, l_graph):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        
        word_represent = self.wordrep(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ## word_embs (batch_size, seq_len, embed_size)
        if self.word_feature_extractor == "CNN":
            batch_size = word_inputs.size(0)
            word_in = torch.tanh(self.word2cnn(word_represent)).transpose(2,1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                if batch_size > 1:
                    cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = cnn_feature.transpose(2,1).contiguous()
        else:
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            ## lstm_out (seq_len, seq_len, hidden_size)
            feature_out = self.droplstm(lstm_out.transpose(1,0))
        
        if gaz_list.size()[1] == 0:
            return self.hidden2tag(feature_out)
        gaz_feature = self.gaz_embeddings(gaz_list)
        gaz_feature = self.gaz_dropout(gaz_feature)
        max_seq_len = feature_out.size()[1]
        gat_input = torch.cat((feature_out, gaz_feature), dim=1)
        gat_feature_1 = self.gat_1(gat_input, t_graph)
        gat_feature_1 = gat_feature_1[:, :max_seq_len, :]
        gat_feature_2 = self.gat_2(gat_input, c_graph)
        gat_feature_2 = gat_feature_2[:, :max_seq_len, :]
        gat_feature_3 = self.gat_3(gat_input, l_graph)
        gat_feature_3 = gat_feature_3[:, :max_seq_len, :]

        ## feature_out (batch_size, seq_len, hidden_size)
        outputs = self.hidden2tag(feature_out)

        if self.strategy == "m":
            crf_feature = torch.cat((outputs, gat_feature_1, gat_feature_2, gat_feature_3), dim=2)
            crf_feature = self.weight(crf_feature)
        elif self.strategy == "v":
            crf_feature = torch.mul(outputs, self.weight1) + torch.mul(gat_feature_1, self.weight2) + torch.mul(
                gat_feature_2, self.weight3) + torch.mul(gat_feature_3, self.weight4)
        else:
            
            crf_feature = self.weight1 * outputs + self.weight4 * gat_feature_3
        #crf_feature = outputs
        return crf_feature

    def sentence_representation(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, ), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ## word_embs (batch_size, seq_len, embed_size)
        batch_size = word_inputs.size(0)
        if self.word_feature_extractor == "CNN":
            word_in = torch.tanh(self.word2cnn(word_represent)).transpose(2,1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                if batch_size > 1:
                    cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = F.max_pool1d(cnn_feature, cnn_feature.size(2)).view(batch_size, -1)
        else:
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            ## lstm_out (seq_len, seq_len, hidden_size)
            ## feature_out (batch_size, hidden_size)
            feature_out = hidden[0].transpose(1,0).contiguous().view(batch_size,-1)
            
        feature_list = [feature_out]
        for idx in range(self.feature_num):
            feature_list.append(self.feature_embeddings[idx](feature_inputs[idx]))
        final_feature = torch.cat(feature_list, 1)
        outputs = self.hidden2tag(self.droplstm(final_feature))
        ## outputs: (batch_size, label_alphabet_size)
        return outputs
