import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from seq2seq.BERT_embedding import BERT_embedding

class FbSeq2seq(nn.Module):

    def __init__(self, encoder_title, encoder, decoder, id2word_dict, decode_function=F.log_softmax):
        super(FbSeq2seq, self).__init__()

        self.decoder = decoder
        self.decode_function = decode_function
        self.id2word = id2word_dict

    def flatten_parameters(self):
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, prev_title, prev_generated_seq=None, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        # The forward pass is going to combine BERT-encoding and decoding together
        bert = BERT_embedding()
        print("#######################Encode title######################################################################")
        vecSen = input_variable.cpu().detach().numpy()  # Convert each sentence from torch tensor to a list of word indices
        batch_size = np.shape(vecSen)[0]
        encoder_outputs, encoder_hidden = bert.encode(vecSen, self.id2word)

        if prev_title is None:
            pt_encoder_states = None
            pt_encoder_hidden = None
        else:
            vecSen = prev_title.cpu().detach().numpy()
            print("#############################################Encode prev title###################################################")
            pt_encoder_states, _ = bert.encode(vecSen, self.id2word)

        if prev_generated_seq is None:
            pg_encoder_states = None
            pg_encoder_hidden = None
        else:
            print("################################################################################Encode prev draft##################")
            vecSen = prev_generated_seq.cpu().detach().numpy()
            batch_size = np.shape(vecSen)[0]                  
            pg_encoder_states, _ = bert.encode(vecSen, self.id2word)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              pt_encoder_states=pt_encoder_states,
                              pg_encoder_states=pg_encoder_states,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
