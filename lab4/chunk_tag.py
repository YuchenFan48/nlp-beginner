from datasets import load_dataset
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchcrf import CRF
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
hidden_dim = 128
epoch = 20
lr = 2e-3
dropout = 0.5

# check this from huggingface
chunk_tags = {'O': 0, 'B-ADJP': 1, 'I-ADJP': 2, 'B-ADVP': 3, 'I-ADVP': 4, 'B-CONJP': 5, 'I-CONJP': 6, 'B-INTJ': 7, 'I-INTJ': 8,
 'B-LST': 9, 'I-LST': 10, 'B-NP': 11, 'I-NP': 12, 'B-PP': 13, 'I-PP': 14, 'B-PRT': 15, 'I-PRT': 16, 'B-SBAR': 17,
 'I-SBAR': 18, 'B-UCP': 19, 'I-UCP': 20, 'B-VP': 21, 'I-VP': 22}


# Use this function to download dataset from huggingface and convert it to a jsonl format
def convert_dataset_to_jsonl(out_dir='Your directory to place the dataset'):
    os.makedirs(out_dir, exist_ok=True)
    dataset = load_dataset("conll2003")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    valid_dataset = dataset["validation"]
    train_dataset_jsonl = []
    test_dataset_jsonl = []
    valid_dataset_jsonl = []
    for train_data, test_data, valid_data in zip(train_dataset, test_dataset, valid_dataset):
        train_dataset_jsonl.append(train_data)
        test_dataset_jsonl.append(test_data)
        valid_dataset_jsonl.append(valid_data)
    with open(out_dir+'/train.jsonl', 'w') as f:
        for line in train_dataset_jsonl:
            f.write(json.dumps(line) + '\n')
    with open(out_dir+'/test.jsonl', 'w') as f:
        for line in test_dataset_jsonl:
            f.write(json.dumps(line) + '\n')
    with open(out_dir+'/valid.jsonl', 'w') as f:
        for line in valid_dataset_jsonl:
            f.write(json.dumps(line) + '\n')
            
def load_data(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def build_vocab(dataset):
    vocab = set()
    for data in dataset:
        tokens = data['tokens']
        for token in tokens:
            if token not in vocab:
                vocab.add(token)
    return vocab

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

def create_glove_embedding_layer(embeddings_dict, vocab):
    matrix_len = len(vocab) + 1  # +1 for <UNK> token

    # the second dimension is the glove dimension. e.g. 50, 100 etc.
    weights_matrix = torch.zeros(
        (matrix_len, len(embeddings_dict[next(iter(embeddings_dict))])))

    word2idx = {"<UNK>": 0}
    weights_matrix[0] = torch.randn(*weights_matrix[0].shape)  # Random embedding for <UNK>

    for i, word in enumerate(vocab, start=1):  # Start enumerating from 1 because 0 is reserved for <UNK>
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

def tokenize_dataset(dataset, word2idx):
    tokenized_dataset = []
    for data in dataset:
        tokens = data['tokens']
        if len(tokens) == 0: # Skip empty sequences
            continue
        input_ids = [word2idx.get(word, word2idx["<UNK>"]) for word in tokens] 
        output = data['chunk_tags']
        assert len(input_ids) == len(output)
        if len(input_ids) > 0 and len(output) > 0:
            tokenized_dataset.append(
                {
                    "input_ids": input_ids,
                    "output": output
                }
            )
    return tokenized_dataset


class NERDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
# Padding input_ids and labels to the same length
def collate_fn(batch):
    input_ids = [torch.tensor(data["input_ids"]) for data in batch]
    labels = [torch.tensor(data["output"]) for data in batch]

    # Ensure there's no empty sequences.
    input_ids = [seq for seq in input_ids if len(seq) > 0]
    labels = [seq for seq in labels if len(seq) > 0]

    input_ids_pad = nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    labels_pad = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    return input_ids_pad, labels_pad


class LSTMCRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, dropout, pretrain_weight=None):
        super(LSTMCRF, self).__init__()

        if pretrain_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrain_weight)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=1)
        
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

        self.crf = CRF(tagset_size, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)  # Apply dropout to embeddings
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)  # Apply dropout to LSTM outputs
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # Masks are used to indicate which tokens are useful
    def loss(self, feats, masks, tags):
        # Negative log likelihood
        return -self.crf(feats, tags, mask=masks, reduction='mean')

    def predict(self, feats, mask=None):
        return self.crf.decode(feats, mask)
    
    
def labels_from_id_to_tag(id_list, id_to_tag):
    return [id_to_tag[id] for id in id_list]

def calculate_f1(predictions, true_labels, id_to_tag):
    y_true = [labels_from_id_to_tag(true, id_to_tag) for true in true_labels]
    y_pred = [labels_from_id_to_tag(pred, id_to_tag) for pred in predictions]
    return f1_score(y_true, y_pred), classification_report(y_true, y_pred)

def train(train_loader, model, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)
        # create a mask, filtering those whose value != 0
        mask = input_ids.ne(0).to(device)

        lstm_feats = model(input_ids)
        loss = model.loss(lstm_feats, mask, labels)
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_train_loss:.4f}")
    
    
def evaluate(dataloader, model, id_to_tag):
    model.eval()
    eval_loss = 0
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            mask = input_ids.ne(0).to(device)
            mask[:, 0] = 1

            # Get LSTM features
            lstm_feats = model(input_ids)

            # Compute the loss
            batch_loss = model.loss(lstm_feats, mask, labels)
            eval_loss += batch_loss.item()

            # Predict the labels
            predictions = model.predict(lstm_feats, mask)

            for preds, trues in zip(predictions, labels):
                preds = preds[:len(trues)]  # truncate predictions to the original sequence length
                all_predictions.append(preds)
                all_true_labels.append(trues.tolist())

    avg_eval_loss = eval_loss / len(dataloader)
    assert len(all_predictions) == len(all_true_labels)

    # Adjust predictions to match the lengths of the true labels
    adjusted_predictions = []
    for pred, true in zip(all_predictions, all_true_labels):
        if len(pred) < len(true):
            adjusted_pred = pred + [chunk_tags['O']] * (len(true) - len(pred))
        else:
            adjusted_pred = pred[:len(true)]
        adjusted_predictions.append(adjusted_pred)

    # Calculate F1 score and accuracy
    f1, report = calculate_f1(adjusted_predictions, all_true_labels, id_to_tag)
    
    # When a full sequence is tagged right, we assume it is correct.
    correct = sum([1 for pred, true in zip(adjusted_predictions, all_true_labels) if pred == true])
    total = len(all_true_labels)
    accuracy = correct / total


    print(f"Evaluation loss: {avg_eval_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(report)

    return avg_eval_loss, accuracy, f1


def pipeline(train_path, test_path, valid_path):
    train_dataset = load_data(train_path)
    test_dataset = load_data(test_path)
    valid_dataset = load_data(valid_path)

    vocab = build_vocab(train_dataset)
    embeddings, word2idx = get_embeddings_and_word2idx(vocab, 'your path to pretrained embeddings')

    tokenized_train_dataset = tokenize_dataset(train_dataset, word2idx)
    tokenized_test_dataset = tokenize_dataset(test_dataset, word2idx)
    tokenized_valid_dataset = tokenize_dataset(valid_dataset, word2idx)

    torch_train_dataset = DataLoader(NERDataset(tokenized_train_dataset), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    torch_test_dataset = DataLoader(NERDataset(tokenized_test_dataset), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    torch_valid_dataset = DataLoader(NERDataset(tokenized_valid_dataset), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = LSTMCRF(
        len(embeddings[0]), 
        hidden_dim=hidden_dim, 
        vocab_size=len(vocab), 
        tagset_size=len(chunk_tags), 
        pretrain_weight=embeddings,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # learning rate decay
    scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
    id_to_tag = {v: k for k, v in chunk_tags.items()}
    best_f1 = 0.0

    for i in range(epoch):
        print(f"Epoch {i + 1}/{epoch}")
        train(torch_train_dataset, model, optimizer)
        scheduler.step() 
        print("Evaluating on Validation Set:")
        val_f1 = evaluate(torch_valid_dataset, model, id_to_tag)[2]
        if float(val_f1) > float(best_f1):
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pth")

    model.load_state_dict(torch.load("best_model.pth"))
    print("Evaluating on Test Set:")
    evaluate(torch_test_dataset, model, id_to_tag)

if __name__ == "__main__":
    pipeline('your path to train_dataset', 'your path to test_datasaet', 'your path to valid_dataset')



