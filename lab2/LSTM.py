import csv
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import nltk
from nltk.corpus import stopwords
import torch.nn
from nltk.tokenize import word_tokenize
import json

nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('english')

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(input_file):
    data_list = []
    # Open the TSV file and read it
    with open(input_file, 'r', encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            data_list.append(row)
    return data_list


def tokenizer(text):
    return word_tokenize(text)


def build_vocab(tokenized_dataset):
    vocab = set()
    for phrase in tokenized_dataset:
        for word in phrase:
            vocab.add(word)
    return vocab

# glove form: word + vector representation


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
    matrix_len = len(vocab)

    # the second dimension is the glove dimension. e.g. 50, 100 etc.
    weights_matrix = torch.zeros(
        (matrix_len, len(embeddings_dict[next(iter(embeddings_dict))])))

    word2idx = {}
    for i, word in enumerate(vocab):
        # i is mapped to word
        try:
            weights_matrix[i] = embeddings_dict[word]
            word2idx[word] = i
        # For words not in glove, using a random number.
        except KeyError:
            weights_matrix[i] = torch.randn(*weights_matrix[i].shape)
            word2idx[word] = i

    embedding = torch.nn.Embedding.from_pretrained(weights_matrix)
    return embedding, word2idx


# vocab is global
def tokenize_dataset(dataset, vocab, glove_path='your path to glove'):
    split_dataset = []
    for data in dataset:
        item = {}
        # get the reasonble word list using jieba
        split_text = tokenizer(data['Phrase'])
        # drop duplicate
        split_text = list(set([
            word for word in split_text if word not in stop_words]))
        label = int(data['Sentiment'])
        item['Phrase'] = split_text
        item['Sentiment'] = label
        split_dataset.append(item)
    # Find all words, including , . and so on.
    glove_embeddings = load_glove_embeddings(glove_path)

    # Each row in embeddings corresponds to a word.
    # Key in word2idx corresponds to a word and the val is index.
    embeddings, word2idx = create_glove_embedding_layer(
        glove_embeddings, vocab)
    tokenized_dataset = []
    for data in split_dataset:
        phrase = data['Phrase']
        label = data['Sentiment']

        # get the number of word in weights matrix.
        input_ids = [word2idx[word] for word in phrase]
        output = label
        tokenized_dataset.append(
            {
                "input_ids": input_ids,
                "label": output
            }
        )
    return embeddings, tokenized_dataset


class TextLSTM(torch.nn.Module):
    def __init__(self, embeddings, embed_size, hidden_size=100, num_classes=5, num_layers=1, dropout=0.2):
        super(TextLSTM, self).__init__()

        self.embedding = embeddings
        self.lstm = torch.nn.LSTM(input_size=embed_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0)

        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # (Batch_size, seq_length, embedding_dim)
        x = self.embedding(x)

        # Pass through LSTM
        # lstm_out: Batch_size x seq_length x hidden_size
        lstm_out, (hidden, _) = self.lstm(x)

        # Use the last hidden state as the representation
        output = self.fc(self.dropout(hidden[-1]))

        return output


def train(model, tokenized_train_dataset, tokenized_test_dataset, optimizer, loss_function, epochs=10, batch_size=32):
    model.train()

    # Convert tokenized data into DataLoader for mini-batch gradient descent
    train_loader = torch.utils.data.DataLoader(
        tokenized_train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    results = []

    for epoch in range(epochs):
        total_loss = 0

        for batch in tqdm(train_loader):
            inputs, labels = batch['input_ids'], batch['label']

            # Move tensors to the configured device
            inputs = torch.LongTensor(inputs).to(device)

            labels = torch.LongTensor(labels).to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f'Epoch: {epoch}, Loss: {total_loss/len(train_loader):.4f}')

        eval_loss, accuracy = evaluate(
            model, tokenized_test_dataset, loss_function, batch_size)
        print(
            f'Epoch: {epoch}, Loss: {round(eval_loss, 4)}, Accuracy: {round(accuracy, 4)}')

        results.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss/len(train_loader), 4),
                "eval_loss": round(eval_loss, 4),
                "acc": round(accuracy, 4)
            }
        )

    return results


def pad_sequence(seq, max_length, padding_value=0):
    return seq + [padding_value] * (max_length - len(seq))


def collate_fn(batch):
    # Collate function to handle variable sequence length in DataLoader
    input_ids = [item['input_ids'] for item in batch]
    label = [item['label'] for item in batch]

    # Determine the max length in this batch
    max_length = max([len(ids) for ids in input_ids])

    # Pad every sequence to the max length
    input_ids_padded = [pad_sequence(ids, max_length) for ids in input_ids]

    return {'input_ids': input_ids_padded, 'label': label}


def evaluate(model, tokenized_test_dataset, loss_function, batch_size=32):
    model.eval()  # set the model to evaluation mode

    test_loader = torch.utils.data.DataLoader(
        tokenized_test_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)

    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['input_ids'], batch['label']

            inputs = torch.LongTensor(inputs).to(device)

            labels = torch.LongTensor(labels).to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            total_loss += loss.item()

            # return in the shape (batch_size, label)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return total_loss / len(test_loader), accuracy


def pipeline(path, out_dir='your out_dir path'):
    os.makedirs(out_dir, exist_ok=True)
    dataset = load_dataset(path)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

    # Build vocab first using the entire dataset
    tokenized_dataset = [tokenizer(data['Phrase']) for data in dataset]
    vocab = build_vocab(tokenized_dataset)

    # Now, tokenize the datasets and get embeddings
    embeddings, tokenized_train_dataset = tokenize_dataset(
        train_dataset, vocab)
    _, tokenized_test_dataset = tokenize_dataset(test_dataset, vocab)

    # it is a pytorch embedding object
    embed_size = embeddings.embedding_dim
    dropout = [0, 0.2, 0.5]
    for d in dropout:
        model = TextLSTM(embeddings, embed_size, dropout=d).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        loss_function = torch.nn.CrossEntropyLoss()

        results = train(model, tokenized_train_dataset,
                        tokenized_test_dataset, optimizer, loss_function)
        result_out_dir = f'{out_dir}/dropout-{d}.json'
        with open(result_out_dir, 'w') as f:
            json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    pipeline(
        'your path to train-dataset')
