# Automated-Story-Generation-with-BERT

## BERT fill & compute BLEU
The latest code is BERT_fill_compute_BLEU_update.py, which incoporates all functions for extracting events, create masked sentences, filling in the blank, score sentence and event to sentence. There is a callable function "BERT_fil()" that takes previous sentences, event and max_masks as input and returns a complete sentence. 

## Editing-Writing-Network with BERT
 Train network: ```python3 main.py --cuda --mode 0```  
 Generate sentence with user input. The user needs to input event and number of drafts in the terminal: ```python3 main.py --cuda --mode 1```  
 Generate sentence: ```python3 main.py --cuda --mode 2```  
 Compute scores: ```python3 main.py --cuda --mode 3```  
 Restart training: ```python3 main.py --cuda --mode 4```
