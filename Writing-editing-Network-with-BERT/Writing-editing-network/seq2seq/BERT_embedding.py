import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

'''
This class is used for getting BERT encoding for a batch of sentences.
We let sentences go through the BERT forward pass one by one 
and use the hidden vectors (usually in the last layer) in BERT as encoding vectors for sentences
'''

class BERT_embedding:
    def __init__(self):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load model
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)

    def vecSen2strSen(self, vecSen, id2word_dict):
        ''' Convert vectorized sentence to a sentence in string form
        Args:
            vecSen(list): A vectorized sentence is a list of word index
            id2word_dict(dictionary): Given the word index, get the corresponding word in the vocab
        '''
        strSen = []
        #print(vecSen)
        for idx in vecSen:
            # Indices 0-3 are for special tokens
            if idx < 4: continue
            strSen.append((id2word_dict[idx]).lower())
        return " ".join(strSen).strip(" <pad>")

    def encode_sen(self, sen):
        '''
        Args:
            sen(str): the input sentence, without period, without CLS or SEP
        '''
        print(sen)
        if len(sen)==0: sen += "."  # In case that the input sentence is empty, we do not want BERT to end up with error
        text = "[CLS] " + sen + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(text)
        
        #print("Number of words: ", len(tokenized_text)-2)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Convert to GPU tokens
        tokens_tensor = tokens_tensor.to(self.device)
        segments_tensors = segments_tensors.to(self.device)
        
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        # encoded_layers have four dimensions: 
        # Number of layers (12)
        # Number of batches (1 sentence)
        # Number of tokens in the sentence
        # Hidden size for each token 
        #print ("Number of layers:", len(encoded_layers))
        layer_i = 0
        #print ("Number of batches:", len(encoded_layers[layer_i]))
        batch_i = 0
        #print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
        token_i = 0
        #print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))
        
        token_embeddings_list = [] 
        # For each token in the sentence...
        for token_i in range(1, len(tokenized_text)-1):
            # Use the hidden ventor in the last hidden layer of BERT
            last_hidden_state = encoded_layers[len(encoded_layers)-1][batch_i][token_i]
            token_embeddings_list.append(last_hidden_state)
        token_embeddings = torch.stack(token_embeddings_list).to(self.device)
        # sentence_embedding: average the second to last layer of all word embeddings
        # with the same size as each token embedding
        sentence_embedding = torch.mean(encoded_layers[11], 1)[0]
        
        return (token_embeddings, sentence_embedding)

    def encode(self, vecSen, id2word_dict):
        '''
        Args:
            vecSen(list): a batch (a list of sentences), each sentence contains max_len number of word ids
            id2word_dict(dictionary): Given the word index, get the corresponding word in the vocab

        '''
        batch_size = np.shape(vecSen)[0]
        encoder_outputs_noPad = []
        encoder_hidden = []
        max_len = 0 
        # max_len could be changed because of BERT's tokenizer
        # So we need to calculate max_len and do padding again after encoding
        for i in range(batch_size):
            strSen = self.vecSen2strSen(vecSen[i], id2word_dict)
            _o, _h = self.encode_sen(strSen)
            if len(_o)>max_len:
                max_len = len(_o)
            encoder_outputs_noPad.append(_o)
            encoder_hidden.append(_h)

        # Do padding according to the new max_len for the batch
        embed_size = encoder_hidden[0].size()[0]
        #print("max_len for batch = ", max_len)
        encoder_outputs = self.pad(encoder_outputs_noPad, max_len, embed_size)
        # convert a list of torch tensors to a tensor
        encoder_outputs = torch.stack(encoder_outputs).to(self.device)
        encoder_hidden = torch.stack(encoder_hidden, 0).to(self.device)
        return (encoder_outputs, encoder_hidden)

    def pad(self, encoder_outputs_noPad, max_len, embed_size):
        for i in range(len(encoder_outputs_noPad)):
            pad = max_len - encoder_outputs_noPad[i].size()[0]
            pad_tensor = torch.zeros([pad, embed_size], dtype=torch.float, device=self.device)
            encoder_outputs_noPad[i] = torch.cat((encoder_outputs_noPad[i], pad_tensor), 0).to(self.device)
        return encoder_outputs_noPad

        
