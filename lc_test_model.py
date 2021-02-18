import pickle
import gzip
import time
import sys
from language_classifier import LanguageClassifier
import matplotlib.pyplot as plt
import numpy as np
from conllu import parse_incr
import sklearn.metrics as skm

# argv = path-to-baseline path-to-best-serialized-model path-to-test-set path-to-test-labels

if __name__ == '__main__':

    baseline = LanguageClassifier()
    with gzip.open(sys.argv[1], 'rb') as fp:
        baseline = pickle.load(fp)
    # print(baseline.most_common_label)

    model = LanguageClassifier()
    with gzip.open(sys.argv[2], 'rb') as fp:
        model = pickle.load(fp)

    f1_scores = []
    precisions = []
    recalls = []
    model_names = ['Baseline', 'Backoff']

    test_data = open(sys.argv[3], "r", encoding="utf-8")
    tokens_list = []
    # creating a list of tokens for each sentence
    for tokens in parse_incr(test_data):
        tokens_list.append(np.array(tokens))
    tokens_list = np.array(tokens_list)
    # extracting only the second column of each entry and making a list of all the tokens
    tokens_list = [[t['form'] for t in token] for token in tokens_list]
    # print(tokens_list)

    labels = np.array(['ewt', 'gum', 'hin'])
    label_data = open(sys.argv[4], "r", encoding="utf-8")
    gold_labels = label_data.readlines()
    gold_labels = np.asarray([label.strip() for label in gold_labels])
    # print(gold_labels)
    gold_labels = [np.where(labels == label)[0][0] for label in gold_labels]

    # baseline
    print("Evaluating Baseline")
    baseline_eval_labels = np.repeat(baseline.most_common_label, len(gold_labels))
    precision = skm.precision_score(gold_labels, baseline_eval_labels, average=None)
    print("Precision per class: {}".format(list(zip(labels, precision))))
    precision = skm.precision_score(gold_labels, baseline_eval_labels, average='macro')
    print("Macro Precision: {}".format(precision))
    precisions.append(precision)
    recall = skm.recall_score(gold_labels, baseline_eval_labels, average=None)
    print("Recall per class: {}".format(list(zip(labels, recall))))
    recall = skm.recall_score(gold_labels, baseline_eval_labels, average='macro')
    print("Macro Recall: {}".format(recall))
    recalls.append(recall)
    f1_score = skm.f1_score(gold_labels, baseline_eval_labels, average=None)
    print("F1 score per class: {}".format(list(zip(labels, f1_score))))
    f1_score = skm.f1_score(gold_labels, baseline_eval_labels, average='macro')
    print("Macro F1 score: {}".format(f1_score))
    f1_scores.append(f1_score)

    print()
    print()

    # best model
    print("Evaluating Best Model")
    model_eval_labels = model.eval(tokens_list)
    precision = skm.precision_score(gold_labels, model_eval_labels, average=None)
    print("Precision per class: {}".format(list(zip(labels, precision))))
    precision = skm.precision_score(gold_labels, model_eval_labels, average='macro')
    print("Macro Precision: {}".format(precision))
    precisions.append(precision)
    recall = skm.recall_score(gold_labels, model_eval_labels, average=None)
    print("Recall per class: {}".format(list(zip(labels, recall))))
    recall = skm.recall_score(gold_labels, model_eval_labels, average='macro')
    print("Macro Recall: {}".format(recall))
    recalls.append(recall)
    f1_score = skm.f1_score(gold_labels, model_eval_labels, average=None)
    print("F1 score per class: {}".format(list(zip(labels, f1_score))))
    f1_score = skm.f1_score(gold_labels, model_eval_labels, average='macro')
    print("Macro F1 score: {}".format(f1_score))
    f1_scores.append(f1_score)

    fig1 = plt.figure()
    
    plt.plot(model_names, precisions, 'b-o', 
             model_names, recalls, 'r-o',
             model_names, f1_scores, 'g-o', alpha=0.5)
    plt.xlabel('Model')
    plt.ylabel('Scores')
    plt.legend(['Precision', 'Recall', 'F1 score'])
    
    plt.tight_layout()
    plt.show()