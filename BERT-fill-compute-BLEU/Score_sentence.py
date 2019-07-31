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
    len1 = len(first)
    len2 = len(second)
    INPUT = first[:]
    INPUT.extend(second)
    indexed_tokens = tokenizer.convert_tokens_to_ids(INPUT)
    segments_ids = []
    segments_ids.extend(np.zeros(len1+len2, np.int32).tolist())
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    predictions = NSPmodel(tokens_tensor, segments_tensors )
    score = predictions[0][0].cpu().detach().numpy()
    # print(score)
    return score


def score_single_sentence(second):
    '''
    Args:
        second(list): A list of tokens, including [SEP] in the end
    Returns:
        score: measures how likely that "second" is a sentence in natural language
    '''
    INPUT = second[:]
    INPUT.insert(0, '[CLS]')
    len2 = len(INPUT)
    indexed_tokens = tokenizer.convert_tokens_to_ids(INPUT)
    segments_ids = []
    segments_ids.extend(np.zeros(len2, np.int32).tolist())
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    predictions = NSPmodel(tokens_tensor, segments_tensors )
    score = predictions[0][0].detach().numpy()
    # print(score)
    return score


def E2S(event, max_masks):
    '''
    Args:
        event(string): A string representing an event, without [CLS], [SEP], or period.
        max_masks(int): The maximum number of masks that is allowed between every two consecutive tokens.
        i.e. at each blank the possible number of masks is in the range [1, max_masks]
    Returns:
        sentences(list): A list of strings, each string is a sentence created by inserting [MASK]s between some tokens of the event in a certain way
    '''
    sentences = []
    event = event.rstrip()
    event = event.split(" ")
    l = len(event)
    for m in product([i for i in range(0,max_masks+1)],repeat = l+1):
        loc=l
        sen = event[:]
        while loc>=0:
            for i in range(m[loc]):
                sen.insert(loc,'[MASK]')
            loc -= 1
        sen.append('.')
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

    # Generated sentences with period in the middle or consecutive repitition of words will be discarded before scoring
    for i in range(len(second)-1):
        if second[i] == second[i+1]:
            second = []
            return (first, second)
    for i in range(len(second)-2):
        if second[i] == '.':
            second = []
            #print(second)
            return (first, second)
    return (first, second)


def extract_whole_sentence(inp):
    '''
    Args:
        inp(string): dataset file name
    Returns:
        ground_truth(list): A list of strings, each representing the ground truth sentence for one training example.
    '''
    ground_truth = []
    with open(inp,'r') as i:
        for line in i.readlines():
            if "<EOS>" in line or "%%%%%" in line:
                continue
            sen = line.split("|||")[2]
            sen = sen.replace(".", "")
            ground_truth.append(sen)
    return (ground_truth)


def extract_raw_event(inp):
    '''
    Args:
        inp(string): dataset file name
    Returns:
        EVENT(list): A list of string, each representing the event of one training example.
        
        Notice that some long sentences are split into several events.
        In this case the corresponding element in EVENT is a concatenation of several events.
        This ensures the one-to-one relationship between events and ground truth sentences.
    '''
    #from pattern.en import conjugate
    event_per_line = []
    vocab = set()
    indv_event = []
    EVENT = []
    with open(inp,'r') as i:
        for line in i.readlines():
            if "<EOS>" in line or "%%%%%" in line:
                continue
            event = line.split("|||")[0]
            events = eval(event)
            # keep track of number of events per line, so that if a long sentence is split into multiple events,
            # those events are combined or merged after BERT, before BLEU
            event_per_line.append(len(events))
            for e in events:
                # if not e[0]=='they':
                    # e[1] = conjugate(e[1], '3sg')
                temp = e[3]
                e[3] = e[4]
                e[4] = temp
                for i in range(len(e)):
                    e[i] = e[i].lower()
                while(e[-1]=='emptyparameter'):
                    e = e[:-1]
                indv_event.append((" ".join(e).replace('emptyparameter', '')))

    e_idx = 0
    for line_idx in range(len(event_per_line)):
        EVENT.append(indv_event[e_idx])
        e_idx += 1
        num_events = event_per_line[line_idx]-1
        while num_events>0:
            EVENT[-1] += " " + indv_event[e_idx]
            e_idx += 1
            num_events -= 1
    return EVENT


'''
Specify Inputs
'''
inputfile = "event_seg.txt"
answerfile = "extracted_wholeSentence.txt"
outputfile = "discard_middle_period&repitition.txt"

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
        sentences = E2S(text, max_masks)
        best_sentence = None
        highest_score = -float('inf')
        print(len(sentences))
        count = 0
        for s in sentences:
            second = tokenizer.tokenize(s)
            first = []
            for i in range(gram-1):
                first.extend(prev[i])
            complete_first, complete_second = fill_in_blanks(first, second)
            if len(complete_second) == 0:
                continue
            count +=1
            score = score_sentence(complete_first, complete_second)
            if score>highest_score:
                highest_score = score
                best_sentence = complete_second
        print(count)
        best_sentence.remove('[SEP]')
        prev.pop(0)
        lengths.pop(0)
        lengths.append(len(best_sentence))
        prev.append(best_sentence)
        
        outputs.append(" ".join(best_sentence))
        print(outputs[-1])


'''
Begin Writing
'''
with open(outputfile, "w") as text_file:
    text_file.write("\n".join(outputs))



