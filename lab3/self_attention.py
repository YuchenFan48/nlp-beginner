import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.nn.functional as F
from ESIM.esim.model import ESIM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import os
os.environ['MKL_VERBOSE'] = '0'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


device = "cuda" if torch.cuda.is_available() else "cpu"
stop_words = stopwords.words('english')

# hyperparameters
hidden_size = 50
dropout = 0.5
num_classes = 3
lr = 4e-4
patience = 4
max_gradient_norm = 10
epoch = 3

def tokenizer(text):
    return word_tokenize(text)


def load_dataset(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def load_glove_embeddings(glove_path):
    embeddings_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddings = torch.tensor([float(val)
                                       for val in values[1:]], dtype=torch.float32)
            embeddings_dict[word] = embeddings
    return embeddings_dict


def build_vocab(tokenized_dataset):
    # special tokens
    # BOS -> Begin of Sentence
    # EOS -> End of Sentence
    # OOV -> Out of Vocab
    vocab = set(['<BOS>', '<EOS>', '<OOV>'])
    for data in tokenized_dataset:
        hypo = data['hypothesis_list']
        prem = data['premise_list']
        for word in hypo:
            if word not in vocab:
                vocab.add(word)
        for word in prem:
            if word not in vocab:
                vocab.add(word)
    return vocab


def create_glove_embedding_layer(embeddings_dict, vocab):
    matrix_len = len(vocab)

    # the second dimension is the glove dimension. e.g. 50, 100 etc.
    weights_matrix = torch.zeros(
        (matrix_len, len(embeddings_dict[next(iter(embeddings_dict))])))

    word2idx = {}
    for i, word in enumerate(vocab):
        if word in embeddings_dict:
            weights_matrix[i] = embeddings_dict[word]
        else:
            # For special tokens and words not in glove, use random embeddings.
            weights_matrix[i] = torch.randn(*weights_matrix[i].shape)
        word2idx[word] = i

    return weights_matrix, word2idx

def get_embeddings_and_word2idx(vocab, glove_path):
    glove_embeddings = load_glove_embeddings(glove_path)
    embeddings, word2idx = create_glove_embedding_layer(glove_embeddings, vocab)
    return embeddings, word2idx

def tokenize_data(dataset, word2idx):
    tokenized_dataset = []
    # map labels to nums
    labels_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    for data in dataset:
        hypo = data['hypothesis_list']
        pre = data['premise_list']
        label = data['label']
        # skip records without label
        if label not in labels_map.keys():
            continue
        hypo_input_ids = [word2idx.get(word, word2idx["<OOV>"]) for word in hypo]  # default to <OOV> if word not in vocab
        pre_input_ids = [word2idx.get(word, word2idx["<OOV>"]) for word in pre]
        output = labels_map[label]
        tokenized_dataset.append({
            "hypo_input_ids": hypo_input_ids,
            "pre_input_ids": pre_input_ids,
            "label": output
        })
    return tokenized_dataset

def preprocess_dataset(dataset):
    info = []
    for data in dataset:
        prem = tokenizer(f'''<BOS> {data['sentence1']} <EOS>''')
        hypo = tokenizer(f'''<BOS> {data['sentence2']} <EOS>''')
        label = data['gold_label']
        info.append({"premise_list": prem, "hypothesis_list": hypo, "label": label})
    return info

# Prepare for the dataset format ESIM needs.
class NLI_Dataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        return (self.tokenized_dataset[idx]['pre_input_ids'], 
                len(self.tokenized_dataset[idx]['pre_input_ids']),
                self.tokenized_dataset[idx]['hypo_input_ids'],
                len(self.tokenized_dataset[idx]['hypo_input_ids']),
                self.tokenized_dataset[idx]['label'])

# Pad to the same length.
def collate_fn(batch):
    premise, premise_lengths, hypothesis, hypothesis_lengths, labels = zip(*batch)
    
    premise = [torch.LongTensor(p) for p in premise]
    hypothesis = [torch.LongTensor(h) for h in hypothesis]
    
    # Pad sequences
    premise = torch.nn.utils.rnn.pad_sequence(premise, batch_first=True)
    hypothesis = torch.nn.utils.rnn.pad_sequence(hypothesis, batch_first=True)
    
    return premise, torch.tensor(premise_lengths), hypothesis, torch.tensor(hypothesis_lengths), torch.tensor(labels)

# Evaluation Metric defined as the ratio of the accuracy of correct classification.
def compute_metrics(preds, labels):
    _, outclass = preds.max(axis=1)
    correct = sum(outclass == labels)
    return correct.item()

def train(epoch, model, train_data_loader, valid_data_loader, optimizer, criterion, max_gradient_norm, scheduler, patience):
    model.train()
    train_loss_list = []
    eval_loss_list = []
    eval_acc_list = []
    best_eval_acc = 0
    
    # Initialize the metrics dictionary to save training progress
    train_metrics = {
        "train_loss": [],
        "eval_loss": [],
        "eval_accuracy": []
    }

    print('----------------------------------Start Training------------------------------')
    for i in range(epoch):
        running_loss = 0
        start_time = time.time()
        print(f'---------------------------------Epoch {i}---------------------------------------')
        
        for data in tqdm(train_data_loader):
            premise, premise_lengths, hypothesis, hypothesis_lengths, labels = data
            premise = premise.to(device)
            premise_lengths = premise_lengths.to(device)
            hypothesis = hypothesis.to(device)
            hypothesis_lengths = hypothesis_lengths.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits, probs = model(premise, premise_lengths, hypothesis, hypothesis_lengths)
            loss = criterion(logits, labels)
            loss.backward()
            # It is used to l2 normalize 
            nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            optimizer.step()
            
            running_loss += loss.item()

        end_time = time.time()
        train_loss = round(running_loss / len(train_data_loader), 4)
        train_loss_list.append(train_loss)
        
        print(f'''
              Epoch: {i}
              Train Loss: {train_loss}
              Train Time: {end_time - start_time}
              ''')

        eval_time, eval_acc, eval_loss = evaluate(model, valid_data_loader, criterion)
        eval_loss_list.append(eval_loss)
        eval_acc_list.append(eval_acc)
        
        # Update the metrics dictionary
        train_metrics["train_loss"].append(train_loss)
        train_metrics["eval_loss"].append(eval_loss)
        train_metrics["eval_accuracy"].append(eval_acc)

        print(f'''
              Epoch: {i}
              Eval Loss: {round(eval_loss, 4)}
              Eval Acc: {round(eval_acc, 4)}
              Best Eval Acc: {round(best_eval_acc, 4)}
              Time: {eval_time}
              ''')

        scheduler.step(eval_acc)
        
        # If the current model is better than the previously saved one, save it
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        if eval_acc < best_eval_acc:
            patience -= 1
        if patience == 0:
            print("--------------------------------Early Stop Here------------------------------")
            break

    # Save the training progress to a JSON file
    with open('results.json', 'w') as file:
        json.dump(train_metrics, file)

        

def evaluate(model, data_loader, criterion):
    model.eval()
    running_loss = 0
    running_correct = 0
    print('--------------------------------------Start Evaluate-----------------------------')
    start_time = time.time()
    for data in tqdm(data_loader):
        premise, premise_lengths, hypothesis, hypothesis_lengths, labels = data
        premise = premise.to(device)
        premise_lengths = premise_lengths.to(device)
        hypothesis = hypothesis.to(device)
        hypothesis_lengths = hypothesis_lengths.to(device)
        labels = labels.to(device)
        logits, probs = model(premise, premise_lengths, hypothesis, hypothesis_lengths)
        loss = criterion(logits, labels)
        running_loss += loss.item()
        running_correct += compute_metrics(probs, labels)
    end_time = time.time()
    return end_time - start_time, running_correct / len(data_loader.dataset), running_loss / len(data_loader)
        



def pipeline(train_path, test_path, valid_path, glove_path):
    train_dataset = load_dataset(train_path)
    test_dataset = load_dataset(test_path)
    valid_dataset = load_dataset(valid_path)
    preprocessed_train_dataset = preprocess_dataset(train_dataset)
    preprocessed_test_dataset = preprocess_dataset(test_dataset)
    preprocessed_valid_dataset = preprocess_dataset(valid_dataset)
    vocab = build_vocab(preprocessed_train_dataset)
    embeddings, word2idx = get_embeddings_and_word2idx(vocab, glove_path)

    tokenized_train_dataset = tokenize_data(preprocessed_train_dataset, word2idx)
    tokenized_test_dataset = tokenize_data(preprocessed_test_dataset, word2idx)
    tokenized_valid_dataset = tokenize_data(preprocessed_valid_dataset, word2idx)
    torch_train_dataset = NLI_Dataset(tokenized_train_dataset)
    train_data_loader = DataLoader(torch_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    torch_test_dataset = NLI_Dataset(tokenized_test_dataset)
    test_dataset_loader = DataLoader(torch_test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    torch_val_dataset = NLI_Dataset(tokenized_valid_dataset)
    val_dataset_loader = DataLoader(torch_val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    model = ESIM(len(vocab),
                len(embeddings[0]),
                hidden_size,
                embeddings=embeddings,
                dropout=dropout,
                num_classes=num_classes,
                device=device).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=patience)
    train(epoch, model, train_data_loader, val_dataset_loader, optimizer, criterion, max_gradient_norm, scheduler, patience)

if __name__ == "__main__":
    pipeline('/Users/apple/Desktop/nlp-fdu/lab3/snli_1.0/snli_1.0_train.jsonl', '/Users/apple/Desktop/nlp-fdu/lab3/snli_1.0/snli_1.0_test.jsonl', '/Users/apple/Desktop/nlp-fdu/lab3/snli_1.0/snli_1.0_dev.jsonl', '/Users/apple/Desktop/nlp-fdu/glove.6B.300d.txt')
