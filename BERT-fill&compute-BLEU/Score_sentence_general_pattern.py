#!/usr/bin/env python
# coding: utf-8


'''
Basic Setup
'''
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM,BertForNextSentencePrediction

# Load pre-trained model (weights)
NSPmodel = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
NSPmodel.eval()

LMmodel = BertForMaskedLM.from_pretrained('bert-base-uncased')
LMmodel.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import torch
import numpy as np
from itertools import zip_longest
from itertools import product

# enable GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LMmodel.to(device)
NSPmodel.to(device)


def score_sentence(first, second):
    ''' 
    Args:
        first(list): A list of tokens, including [CLS] in the front
        second(list): A list of tokens, including [SEP] in the end
    Returns:
        score: measures how likely that "second" is the next sentence of "first"
    '''
    first.append('[SEP]')
    len1 = len(first)
    len2 = len(second)
    INPUT = first[:]
    INPUT.extend(second)
    indexed_tokens = tokenizer.convert_tokens_to_ids(INPUT)
    segments_ids = []
    #segments_ids.extend(np.zeros(len1+len2, np.int32).tolist())
    segments_ids.extend(np.zeros(len1, np.int32).tolist())
    segments_ids.extend(np.ones(len2, np.int32).tolist())
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    predictions = NSPmodel(tokens_tensor, segments_tensors )
    score = predictions[0][0].cpu().detach().numpy()
    # print(score)
    return score


def general_pattern_2S(event, max_masks):
    '''
    Args:
        event(string): A string representing a general pattern sentence, without [CLS], [SEP], or period.
        max_masks(int): The maximum number of masks that is allowed at each blank.
        i.e. at each blank the possible number of masks is in the range [1, max_masks]
    Returns:
        sentences(list): A list of strings, each string is a sentence created by inserting [MASK]s at some blanks in a certain way
    ''' 
    sentences = []
    event = event.rstrip()
    event = event.split(" ")
    # print(event)
    l = len(event)
    index = []
    for i in range(l):
        if event[i]=='[MASK]':
            index.append(i)
    for m in product([i for i in range(0,max_masks)],repeat = len(index)):
        loc_ind=len(index)-1
        sen = event[:]
        while loc_ind>=0:
            loc = index[loc_ind]
            for i in range(m[loc_ind]):
                sen.insert(loc,'[MASK]')
            loc_ind -= 1
        #sen.append('.')
        sentences.append(" ".join(sen))
    return sentences


def fill_in_blanks(first, second):
    '''
    Args:
        first(list): A list of tokens, with period and without [CLS], which is a complete sentence.
        second(list): A list of tokens, with period and without [SEP], which may contain several [MASK]s.
    Returns:
        first(list): A list of tokens which form a complete sentence, including [CLS] in the front.
        second(list): A list of tokens which represent the complete sentence after BERT replacing those [MASK]s in the input with real words, including [SEP] in the end.

        The resturned arguments are ready to be passed into "score_sentence" function.
    '''    
    first.insert(0, '[CLS]')
    second.append('[SEP]')
    len1 = len(first)
    len2 = len(second)
    INPUT = first[:]
    INPUT.extend(second)
    index = []
    for i in range(len1-1, len(INPUT)):
        if INPUT[i]=='[MASK]':
            index.append(i)
    segments_ids = []
    segments_ids.extend(np.zeros(len(INPUT), np.int32).tolist())
        
    indexed_tokens = tokenizer.convert_tokens_to_ids(INPUT)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to(device)
    segments_tensor = segments_tensor.to(device)
    with torch.no_grad():
        predictions = LMmodel(tokens_tensor, segments_tensor)
    for i in index:
        predicted_index = torch.argmax(predictions[0, i]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        second[i-len1] = predicted_token[0]
    second = [i for i, j in zip_longest(second, second[1:]) if i != j]    
    return (first, second)


'''
Specify Inputs
'''
inputfile = "general_pattern_seg.txt"
answerfile = "extracted_wholeSentence.txt"
outputfile = "BERT_search_general_pattern_maxmask3_30_remove_rep.txt"

prev = []
prev.append(['the', 'talks', 'stalled', 'again','.'])
prev.append(['at', 'gate', 'daniel','plans', 'to', 'go', 'to', 'antarctica', 'but', 'then', 'weir', 'enters', 'and', 'tells', 'him','.'])
prev.append(['he', 'gets', 'angry', 'about', 'the', 'situation','.'])
prev.append(['she', 'cannot', 'do', 'anything','.'])
prev.append(['she', 'leaves','.'])
prev.append(['they', 'are', 'not', 'allowed', 'to', 'use', 'the', 'place', 'one','.'])
prev.append(['she', 'later', 'talks', 'with', 'someone', 'and', 'its', 'members','protest','.'])
prev.append(['they', 'cannot', 'use', 'it', 'because', 'they', 'want', 'to', 'show', 'their', 'goodwill', 'to', 'the', 'world','.'])
prev.append(['she', 'tells', 'them','.'])

prev.append(['carter', 'then', 'proposes', 'using', 'the', 'modified', 'it', 'to', 'get', 'to', 'there', 'to', 'contact', 'him', 'who', 'could', 'help', 'jack','.'])
prev.append(['they', 'perhaps', 'need', 'that', 'ship', 'in', 'the', 'future', 'to', 'defend', 'off', 'the','.'])
prev.append(['the', 'engines', 'could', 'burn', 'out', 'on', 'the', 'flight','.'])
prev.append(['however', 'weir', 'denies', 'request','.'])
prev.append(['later', 'carter', 'talks', 'with', 'weir', 'privately', 'and', 'asks', 'her', 'to', 'consider', 'her', 'request','.'])
prev.append(['she', 'denies','.'])
prev.append(['carter', 'threatens', 'to', 'refuse', 'to', 'work', 'on', 'the', 'modified', 'cargo', 'ship','.'])
prev.append(['she', 'gets', 'the', 'allowance','.'])
prev.append(['carter', 'then', 'talks', 'with', 'him', 'about', 'their', 'flight','.'])
prev.append(['daniel', 'enters','.'])

prev.append(['he', 'has', 'to', 'stay', 'because', 'if', 'the', 'two', 'fail', 'he', 'would', 'be', 'the', 'only', 'one', 'left', 'to', 'help','.'])
prev.append(['he', 'is', 'informed','.'])
prev.append(['some', 'time', 'later', 'carter', 'and', 'tim', 'fly', 'through', 'hyper','space', 'to', 'mars','.'])
prev.append(['neill', 'modified', 'the', 'engines','.'])
prev.append(['he', 'did', 'it','.'])
prev.append(['during', 'the', 'flight', 'carter', 'tries', 'to', 'find', 'out', 'but', 'she', 'cannot', 'find', 'out','.'])
prev.append(['she', 'then', 'tries', 'to', 'start', 'a', 'conversation', 'with', 'tim','.'])
prev.append(['suddenly', 'the', 'gate', 'is', 'activated','.'])
prev.append(['they', 'receive', 'a', 'text', 'message', 'from', 'carol',',', 'a', 'system', 'lord', 'who', 'wants', 'to', 'arrange', 'a', 'meeting', 'between', 'earth', 'and', 'the', 'system', 'lords','.'])
prev.append(['weir', 'is', 'then', 'authorized', 'by', 'president', 'henry', 'hayes', 'to', 'start', 'negotiations','.'])

lengths = []
gram = 30
for i in range(gram-1):
    lengths.append(len(prev[i]))

outputs= []
max_masks = 2


'''
Begin BERT Filling!
'''
with open(inputfile,'r') as i:
    for text in i.readlines():
        sentences = general_pattern_2S(text, max_masks)
        best_sentence = None
        highest_score = -float('inf')
        print(len(sentences))
        for s in sentences:
            second = tokenizer.tokenize(s)
            first = []
            for i in range(gram-1):
                first.extend(prev[i])
            complete_first, complete_second = fill_in_blanks(first, second)
            score = score_sentence(complete_first, complete_second)
            if score>highest_score:
                highest_score = score
                best_sentence = complete_second
        
        best_sentence.remove('[SEP]')
        prev.pop(0)
        lengths.pop(0)
        lengths.append(len(best_sentence))
        prev.append(best_sentence)
        
        outputs.append(" ".join(best_sentence))
        print("finish line")


'''
Begin Writing!
'''
with open(outputfile, "w") as text_file:
    text_file.write("\n".join(outputs))


