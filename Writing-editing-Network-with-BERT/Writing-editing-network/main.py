import time, argparse, math, os, sys, pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from torch.backends import cudnn
from utils import Vectorizer, headline2abstractdataset
from seq2seq.fb_seq2seq import FbSeq2seq
from seq2seq.EncoderRNN import EncoderRNN
from seq2seq.DecoderRNNFB import DecoderRNNFB
from predictor import Predictor
from pprint import pprint
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
sys.path.insert(0,'..')
from eval import Evaluate


class Config(object):
    cell = "GRU"
    emsize = 768
    nlayers = 1
    lr = 0.001
    epochs = 30 #10
    batch_size = 128 #240
    dropout = 0
    bidirectional = False
    relative_data_path = '/data/train0.txt'
    relative_dev_path = '/data/valid0.txt'
    relative_gen_path = '/data/update_gen2tt%d.txt'
    relative_img_path = '/img/'
    relative_score_path = '/data/score%d.txt'
    max_grad_norm = 10
    min_freq = 1 #5
    num_exams = 3


class ConfigTest(object):
    cell = "GRU"
    emsize = 3
    nlayers = 1
    lr = 1
    epochs = 3
    batch_size = 2
    dropout = 0
    bidirectional = True
    relative_data_path = '/data/baby_train.txt'
    relative_dev_path = '/data/valid0.txt'
    relative_gen_path = '/data/fake%d.dat'
    max_grad_norm = 1
    min_freq = 0
    num_exams = 3


cudnn.benchmark = True
parser = argparse.ArgumentParser(description='seq2seq model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str,  default='params.pkl',
                    help='path to save the final model')
parser.add_argument('--mode', type=int,  default=0,
                    help='train(0)/predict_sentence(1)/predict_file(2) or evaluate(3)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

config = Config()
#config = ConfigTest()

cwd = os.getcwd()
data_path = cwd + config.relative_data_path
img_path = cwd + config.relative_img_path
vectorizer = Vectorizer(min_frequency=config.min_freq)
abstracts = headline2abstractdataset(data_path, vectorizer, args.cuda, raw_event=True, max_len=1000)
dev_data_path = cwd + config.relative_dev_path
abstracts1 = headline2abstractdataset(dev_data_path, abstracts.vectorizer, args.cuda, raw_event = True, max_len=1000)
print("number of training examples: %d" % len(abstracts))
print("max_len in dataset: ", abstracts.max_len)
vocab_size = abstracts.vectorizer.vocabulary_size
embedding = nn.Embedding(vocab_size, config.emsize, padding_idx=0)
encoder_title = EncoderRNN(vocab_size, embedding, abstracts.head_len, config.emsize, input_dropout_p=config.dropout,
                     n_layers=config.nlayers, bidirectional=config.bidirectional, rnn_cell=config.cell)
encoder = EncoderRNN(vocab_size, embedding, abstracts.abs_len, config.emsize, input_dropout_p=config.dropout, variable_lengths = False,
                  n_layers=config.nlayers, bidirectional=config.bidirectional, rnn_cell=config.cell)
decoder = DecoderRNNFB(vocab_size, embedding, abstracts.abs_len, config.emsize, sos_id=2, eos_id=1,
                     n_layers=config.nlayers, rnn_cell=config.cell, bidirectional=config.bidirectional,
                     input_dropout_p=config.dropout, dropout_p=config.dropout)
model = FbSeq2seq(encoder_title, encoder, decoder, abstracts.vectorizer.idx2word)

criterion = nn.CrossEntropyLoss(ignore_index=0)
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

# Mask variable
def _mask(prev_generated_seq):
    prev_mask = torch.eq(prev_generated_seq, 1)
    lengths = torch.argmax(prev_mask,dim=1)
    max_len = prev_generated_seq.size(1)
    mask = []
    for i in range(prev_generated_seq.size(0)):
        if lengths[i] == 0:
            mask_line = [0] * max_len
        else:
            mask_line = [0] * lengths[i].item()
            mask_line.extend([1] * (max_len - lengths[i].item()))
        mask.append(mask_line)
    mask = torch.ByteTensor(mask)
    if args.cuda:
        mask = mask.cuda()
    return prev_generated_seq.data.masked_fill_(mask, 0)

def train_batch(input_variable, prev_title, input_lengths, target_variable, model,
                teacher_forcing_ratio):
    loss_list = []
    # Forward propagation
    prev_generated_seq = input_variable  # We consider that "draft 0" is the same as title
    target_variable_reshaped = target_variable[:, 1:].contiguous().view(-1)

    for i in range(config.num_exams):
        decoder_outputs, _, other = \
            model(input_variable, prev_title, prev_generated_seq, input_lengths,
                   target_variable, teacher_forcing_ratio)

        decoder_outputs_reshaped = decoder_outputs.view(-1, vocab_size)
        lossi = criterion(decoder_outputs_reshaped, target_variable_reshaped)
        loss_list.append(lossi.item())
        model.zero_grad()
        lossi.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        prev_generated_seq = torch.squeeze(torch.topk(decoder_outputs, 1, dim=2)[1]).view(-1, decoder_outputs.size(1))
        prev_generated_seq = _mask(prev_generated_seq)
    return loss_list

def train_epoches(dataset, model, n_epochs, abstracts1, teacher_forcing_ratio):
    train_loader = DataLoader(dataset, config.batch_size)
    model.train(True)
    print("Number of batches: ", len(train_loader))
    prev_epoch_loss_list = [100] * config.num_exams
    batch_losses = []
    epoch_losses = []
    valid_avg_scores = []
    for epoch in range(1, n_epochs + 1):
        torch.set_grad_enabled(True)
        model.train(True)
        epoch_examples_total = 0
        epoch_loss_list = [0] * config.num_exams
        batch_loss_for_this_epoch = []
        prev_title = None
        for batch_idx, (source, target, input_lengths) in enumerate(train_loader):
            if batch_idx>-1:
                input_variables = source
                target_variables = target
                # train model
                loss_list = train_batch(input_variables, prev_title, input_lengths.tolist(),
                               target_variables, model, teacher_forcing_ratio)
                prev_title = source
                # Record average loss
                num_examples = len(source)
                batch_loss_for_this_epoch.append(loss_list)
                epoch_examples_total += num_examples
                for i in range(config.num_exams):
                    epoch_loss_list[i] += loss_list[i] * num_examples
                print("epoch ", epoch, ", batch ", batch_idx)
                if batch_idx % 10 == 0:
                        torch.save(model.state_dict(), args.save)
        print("Finish training !!!")
        for i in range(config.num_exams):
            epoch_loss_list[i] /= float(epoch_examples_total)
        
        log_msg = "Finished epoch %d with losses:" % epoch
        print(log_msg)
        pprint(epoch_loss_list)

        valid_score = validate(epoch, abstracts1)

        # Plot loss curve with regard to the batch indices for this epoch
        batch_losses.append(batch_loss_for_this_epoch)
        plot_loss(batch_loss_for_this_epoch, "Epoch "+str(epoch))
        epoch_losses.append(epoch_loss_list)

        # Decide whether to early stop
        if prev_epoch_loss_list[:-1] < epoch_loss_list[:-1]:
            break
        elif len(valid_avg_scores)>0 and valid_avg_scores[-1] > valid_score:
            break
        else:
            prev_epoch_loss_list = epoch_loss_list[:]
            valid_avg_scores.append(valid_score)

    return (batch_losses, epoch_losses, valid_avg_scores)


def plot_loss(losses, title = ""):
    for exam_idx in range(len(losses[0])):
        x = [i+1 for i in range(len(losses))]
        y = [i[exam_idx] for i in losses]
        plt.plot(x, y)
    plt.show()
    plt.savefig(img_path + title)
    plt.close('all')

def plot_epoch_loss(epoch_losses):
    plot_loss(epoch_losses, "epoch_loss")

def plot_valid_score(scores):
    plot_loss(scores, "validation average scores")


def validate(epoch_idx, abstracts1):
    model.load_state_dict(torch.load(args.save))
    print("model restored")
    test_loader = DataLoader(abstracts1, config.batch_size)
    eval_f = Evaluate()
    num_exams = 3
    predictor = Predictor(model, abstracts1.vectorizer)
    print("Start Evaluating")
    print("Test Data: ", len(abstracts1))
    cand, ref = predictor.preeval_batch(test_loader, len(abstracts1), num_exams)
    scores = []
    final = []
    fields = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L"]
    for i in range(6):
        scores.append([])
    for i in range(num_exams):
        print("No.", i)
        final_scores = eval_f.evaluate(live=True, cand=cand[i], ref=ref)
        for j in range(6):
            scores[j].append(final_scores[fields[j]])
    with open('figure.pkl', 'wb') as f:
        pickle.dump((fields, scores), f)
    # Start writing ...
    f_out_name = cwd + config.relative_score_path
    f_out_name = f_out_name % epoch_idx
    f_out = open(f_out_name, 'w')
    for j in range(6):
        f_out.write(fields[j] + ':  ')
        f_out.write(str(final_scores[fields[j]]) + '\n')
        final.append(final_scores[fields[j]])
    f_out.close()
    print("FFFF = ", final_scores)
    return sum(final) / float(len(final))


if __name__ == "__main__":
    if args.mode == 0:
        # train
        try:
            print("start training...")
            batch_losses, epoch_losses, valid_avg_scores = train_epoches(abstracts, model, config.epochs, abstracts1, teacher_forcing_ratio=0.5)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        torch.save(model.state_dict(), args.save)
        print("model saved")
        plot_epoch_loss(epoch_losses)
        #plot_valid_score(valid_avg_scores)
    elif args.mode == 1:
        # predict sentence
        model.load_state_dict(torch.load(args.save))
        print("model restored")
        predictor = Predictor(model, abstracts.vectorizer)
        while True:
            seq_str = input("Type in a source sequence:\n")
            seq = seq_str.strip().split(' ')
            num_exams = int(input("Type the number of drafts:\n"))
            print("\nresult:")
            outputs = predictor.predict(seq, num_exams)
            for i in range(num_exams):
                print(i)
                print(outputs[i])
            print('-'*120)
    elif args.mode == 2:
        num_exams = 3
        # predict file
        model.load_state_dict(torch.load(args.save))
        print("model restored")
        predictor = Predictor(model, abstracts.vectorizer)
        data_path = cwd + config.relative_dev_path
        abstracts = headline2abstractdataset(data_path, abstracts.vectorizer, args.cuda, raw_event=True, max_len=1000)
        print("number of test examples: %d" % len(abstracts))
        f_out_name = cwd + config.relative_gen_path
        outputs = []
        title = []
        for j in range(num_exams):
            outputs.append([])
        i = 0
        print("Start generating:")
        
        train_loader = DataLoader(abstracts, config.batch_size)
        prev_title = None
        for batch_idx, (source, target, input_lengths) in enumerate(train_loader):
            
            output_seq = predictor.predict_batch(source, prev_title, input_lengths.tolist(), num_exams)
            for seq in output_seq:
                title.append(seq[0])
                for j in range(num_exams):
                    outputs[j].append(seq[j+1])
                i += 1
                if i % 100 == 0:
                    print("Percentages:  %.4f" % (i/float(len(abstracts))))
            prev_title = source

        print("Start writing:")
        
        for i in range(num_exams):
            out_name = f_out_name % i
            f_out = open(out_name, 'w')
            for j in range(len(title)):
                f_out.write(title[j] + '\n' + outputs[i][j] + '\n\n')
                if j % 100 == 0:
                    print("Percentages:  %.4f" % (j/float(len(abstracts))))
            f_out.close()
        f_out.close()

    elif args.mode == 3:
        validate(0, abstracts1)

    elif args.mode == 4:
        # predict sentence
        model.load_state_dict(torch.load(args.save))
        print("model restored")
        # train
        try:
            print("Resume training...")
            batch_losses, epoch_losses, valid_avg_scores = train_epoches(abstracts, model, config.epochs, abstracts1, teacher_forcing_ratio=1)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        torch.save(model.state_dict(), args.save)
        print("model saved")
        plot_epoch_loss(epoch_losses)
        #plot_valid_score(valid_avg_scores)
