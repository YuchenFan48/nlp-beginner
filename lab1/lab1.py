import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import time
import os
import json
import pickle
from random import shuffle

'''
We define that
label y should be in the form of (y1, y2...yi...).
feature x should be displayed in the row form that a row is a piece of data.
Presume we have n records, m features and c labels.
y -> (n, c)
x -> (n, m)
y = x.dot(w)
w should be (m, c)
'''


def load_dataset(input_file):
    data_list = []
    # Open the TSV file and read it
    with open(input_file, 'r', encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            data_list.append(row)
    return data_list


def BOW_feature(dataset):
    feature_set = set()
    for row in dataset:
        token_list = row['Phrase'].split()
        for token in token_list:
            feature_set.add(token)
    return feature_set


def embedding_using_BOW_feature(dataset, feature_set):
    feature_size = len(feature_set)
    feature_list = np.zeros((len(dataset), feature_size))
    feature_map = dict(zip(feature_set, range(feature_size)))
    for i, row in enumerate(dataset):
        token_list = row['Phrase'].split()
        for token in token_list:
            if token in feature_set:
                feature = feature_map[token]
                feature_list[i][feature] += 1
    return np.array(feature_list)


# It is a character-level n-gram because word-level contains too much vector space.
# If you possess enough CPU space or GPU, you can modify it into word-level ngram.
def ngram_feature(data, ngram=2):
    # ngram特征集合
    feature_set = set()
    for row in data:
        token_list = row['Phrase'].split()
        # 对每个词提取ngram特征
        for token in token_list:
            for i in range(len(token)-ngram):
                feature_set.add(token[i:i+ngram])
    return feature_set


def embedding_using_ngram_feature(data, feature_set, ngram=2):
    feature_size = len(feature_set)
    feature_list = np.zeros((data.shape[0], feature_size)).astype('int16')
    feature_map = dict(zip(feature_set, range(feature_size)))
    for index in range(data.shape[0]):
        token_list = data[index]['Phrase'].split()
        for token in token_list:
            for i in range(len(token)-ngram):
                gram = token[i:i+ngram]
                if gram in feature_set:
                    feature_index = feature_map[gram]
                    feature_list[index, feature_index] += 1
    return feature_list


def get_features(method, dataset, gram):
    train_dataset, test_dataset = train_test_split(dataset)
    if method == 'word':
        feature_set = BOW_feature(dataset)
        train_feature_list = np.array(
            embedding_using_BOW_feature(train_dataset, feature_set))
        test_feature_list = np.array(
            embedding_using_BOW_feature(test_dataset, feature_set))
    elif method == 'ngram':
        feature_set = ngram_feature(dataset, gram)
        train_feature_list = np.array(
            embedding_using_ngram_feature(train_dataset, feature_set, gram))
        test_feature_list = np.array(
            embedding_using_ngram_feature(test_dataset, feature_set, gram))
    return train_dataset, test_dataset, train_feature_list, test_feature_list


def train(train_X, train_y, test_X, test_y, batch_size=32, lr=1e0, epoch=100, class_number=5, loss='softmax'):
    results = {}
    train_iter = train_X.shape[0] // batch_size
    top_1_weights = []
    weights = np.zeros((train_X.shape[1], class_number))
    train_loss_list = []
    test_loss_list = []
    for i in range(epoch):
        start_time = time.time()
        print(f"-------------------Epoch{i}----------------")
        train_loss = 0
        test_loss = 0
        for j in tqdm(range(train_iter)):
            feature = train_X[j * batch_size: (j + 1) * batch_size]
            label = train_y[j * batch_size: (j + 1) * batch_size]
            # Look it up in the softmax regression and logistic regression formula.
            if loss == 'softmax':
                y = np.exp(feature.dot(weights))
                y_hat = y / np.sum(y, axis=1).reshape(-1, 1)
                train_loss = -1/batch_size * np.sum(label * np.log(y_hat))
                weights += lr * (feature.T.dot(label - y_hat)) / batch_size
            else:
                y_hat = 1 / (1 + np.exp(-feature.dot(weights)))
                train_loss = -1/batch_size * \
                    np.sum(label * np.log(y_hat) +
                           (1 - label) * np.log(1 - y_hat))
                weights += lr * (feature.T.dot(label - y_hat)) / batch_size
        train_loss_list.append(train_loss)

        # We use the acc as the metric to define the best model.
        if loss == 'softmax':
            y_test = np.exp(test_X.dot(weights))
            y_hat_test = y_test / np.sum(y_test, axis=1).reshape(-1, 1)
            test_loss = -1/test_X.shape[0] * \
                np.sum(test_y * np.log(y_hat_test))
        else:
            y_hat_test = 1 / (1 + np.exp(-test_X.dot(weights)))
            test_loss = -1/test_X.shape[0] * np.sum(test_y * np.log(
                y_hat_test) + (1 - test_y) * np.log(1 - y_hat_test))
        y_hat_test = y_hat_test / np.sum(y_hat_test, axis=1)[:, np.newaxis]
        y_pred = np.array([np.argmax(i) for i in y_hat_test])
        y_label = np.array([np.argmax(i) for i in test_y])
        acc = np.sum(y_label.astype('int16') ==
                     y_pred.astype('int16')) / y_label.shape[0]

        end_time = time.time()
        results[str(i)] = {}
        results[str(i)] = {
            'train_loss': round(train_loss, 4),
            'test_loss': round(test_loss, 4),
            'run_time': end_time - start_time,
            'acc': round(acc, 4)
        }

        # compare acc
        if len(top_1_weights) == 0 or acc >= max([item[2] for item in top_1_weights]):
            top_1_weights.append((i, test_loss, acc, weights.copy()))
        top_1_weights = sorted(
            top_1_weights, key=lambda x: x[2], reverse=True)[:1]
        test_loss_list.append(test_loss)
        log = f'''
        Accuracy: {round(acc, 4)}
        Train_loss: {round(train_loss, 4)}
        Test_loss: {round(test_loss, 4)}
        '''
        print(log)
    return results, top_1_weights


# Modify label into a one-hot label
def get_one_hot_label(dataset):
    labels = [data['Sentiment'] for data in dataset]
    class_number = len(set(labels))
    one_hot_label = np.zeros((len(labels), class_number))
    for i, label in enumerate(labels):
        one_hot_label[i][int(label)] += 1
    return one_hot_label


def train_test_split(dataset, ratio=0.8):
    train_dataset = np.array(dataset[:int(len(dataset) * ratio)])
    test_dataset = np.array(dataset[int(len(dataset) * ratio):])
    return train_dataset, test_dataset


def pipeline(out_dir='/Users/apple/Desktop/nlp-fdu/lab1'):
    global_best_acc = 0
    best_weights = []
    best_parameter = {}

    os.makedirs(out_dir, exist_ok=True)
    dataset = load_dataset(
        '/Users/apple/Desktop/nlp-fdu/sentiment-analysis-on-movie-reviews/train.tsv')
    shuffle(dataset)

    parameter_list = [
        {"method": "ngram", "batch_size": 32, "lr": 1e-1,
            "epoch": 30, "loss": "logistic", "gram": 4},
        {"method": "ngram", "batch_size": 32, "lr": 1e0,
            "epoch": 30, "loss": "softmax", "gram": 4},
        {"method": "word", "batch_size": 32, "lr": 1e0,
            "epoch": 30, "loss": "logistic", "gram": 1},
        {"method": "ngram", "batch_size": 32, "lr": 1e0,
            "epoch": 30, "loss": "logistic", "gram": 4},
        {"method": "word", "batch_size": 32, "lr": 1e-1,
            "epoch": 30, "loss": "softmax", "gram": 1},
        {"method": "word", "batch_size": 32, "lr": 1e-1,
            "epoch": 30, "loss": "logistic", "gram": 1},
        {"method": "word", "batch_size": 32, "lr": 1e0,
            "epoch": 30, "loss": "softmax", "gram": 1},
        {"method": "ngram", "batch_size": 32, "lr": 1e-1,
            "epoch": 30, "loss": "softmax", "gram": 4},
    ]

    for parameter in parameter_list:
        parameter_acc = 0

        # Extract parameters
        method = parameter['method']
        batch_size = parameter['batch_size']
        lr = parameter['lr']
        epoch = parameter['epoch']
        loss = parameter['loss']
        ngram = parameter['gram']

        train_dataset, test_dataset, train_X, test_X = get_features(
            method, dataset, ngram)
        train_y, test_y = get_one_hot_label(
            train_dataset), get_one_hot_label(test_dataset)

        results, best_3_weights = train(
            train_X, train_y, test_X, test_y, batch_size=batch_size, lr=lr, epoch=epoch, class_number=5, loss=loss)

        parameter_out_dir = f'{out_dir}/{method}-{lr}-{loss}'
        os.makedirs(parameter_out_dir, exist_ok=True)
        results_out_dir = f'{parameter_out_dir}/{method}-{lr}-{loss}-log.json'

        with open(results_out_dir, 'w') as f:
            json.dump(results, f, ensure_ascii=False)

        new_acc = best_3_weights[0][2]
        if new_acc > parameter_acc:
            parameter_acc = new_acc

            # Check if this parameter's best model is the global best
            if parameter_acc > global_best_acc:
                global_best_acc = parameter_acc
                best_weights = best_3_weights[0][3]
                best_parameter = parameter

        # Save model weights for this parameter
        cur_epoch = best_3_weights[0][0]
        model_out_dir = f'{parameter_out_dir}/{method}-{lr}-{loss}-{cur_epoch}-checkpoint.pkl'
        with open(model_out_dir, 'wb') as file:
            pickle.dump(best_3_weights[0][3], file)

    return best_weights, best_parameter


def eval(weights, eval_data_path, train_data_path, best_parameter, out_dir='/Users/apple/Desktop/nlp-fdu/sentiment-analysis-on-movie-reviews/test.csv'):
    eval_dataset = load_dataset(eval_data_path)
    train_dataset = load_dataset(train_data_path)
    eval_dataset = np.array(eval_dataset)
    best_metric = best_parameter['method']
    gram = best_parameter['gram']
    if best_metric == 'ngram':
        feature_set = ngram_feature(train_dataset, ngram=gram)
        eval_feature_list = embedding_using_ngram_feature(
            eval_dataset, feature_set, ngram=gram)
        y_test = np.exp(eval_feature_list.dot(weights))
        y_hat_test = y_test / np.sum(y_test, axis=1).reshape(-1, 1)
    else:
        feature_set = BOW_feature(train_dataset)
        eval_feature_list = embedding_using_BOW_feature(
            eval_dataset, feature_set)
        y_hat_test = 1 / (1 + np.exp(eval_feature_list.dot(weights)))
    y_pred = np.array([np.argmax(i) for i in y_hat_test])
    phrase_ids = [item['PhraseId'] for item in eval_dataset]
    phrases = [item['Phrase'] for item in eval_dataset]

    df = pd.DataFrame({
        'PhraseId': phrase_ids,
        'Phrase': phrases,
        'Prediction': y_pred
    })
    df.to_csv(out_dir)


if __name__ == "__main__":
    weights, best_parameter = pipeline()
    eval(weights, '/Users/apple/Desktop/nlp-fdu/sentiment-analysis-on-movie-reviews/test.tsv',
         '/Users/apple/Desktop/nlp-fdu/sentiment-analysis-on-movie-reviews/train.tsv', best_parameter)
