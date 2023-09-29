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


class TextCNN(torch.nn.Module):
    def __init__(self, embeddings, embed_size, num_classes, kernel_sizes, num_channels, dropout=0.2):
        super(TextCNN, self).__init__()

        self.embedding = embeddings

        # 1-dimension CNN used for text.
        # CNN layer -> FC layer -> Activation function
        self.conv1d_list = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels=embed_size,
                            out_channels=num_channels,
                            kernel_size=k)
            for k in kernel_sizes
        ])

        self.fc = torch.nn.Linear(
            len(kernel_sizes) * num_channels, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)

        # standard CNN input (batch size, num channels, sequence length)
        # input form (batch size, sequence length, num channels)
        x = x.transpose(1, 2)

        x_list = [self.relu(conv(x)) for conv in self.conv1d_list]

        # compress the feature sequence into a scalar, which is the most significant feature.
        x_list = [torch.nn.functional.max_pool1d(
            x_, kernel_size=x_.size(2)) for x_ in x_list]

        x = torch.cat(x_list, 2)

        # reshape as (batch size, num of features)
        x = x.view(x.size(0), -1)

        logits = self.fc(x)

        return logits


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


def pipeline(path, out_dir='your out_put dir'):
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

    # kernel_size is the size of the receptive field.
    kernel_sizes = [2, 3, 4]

    # hyperparameter
    num_channels = 100

    # it is a pytorch embedding object
    embed_size = embeddings.embedding_dim
    num_classes = 5
    dropout = [0, 0.2, 0.5]
    for d in dropout:
        model = TextCNN(embeddings, embed_size, num_classes,
                        kernel_sizes, num_channels, d).to(device)

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
