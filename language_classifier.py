import numpy as np
import sys
import time
from conllu import parse_incr
from model import NgramLanguageModel
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sklearn.metrics as skm
np.seterr(all='ignore')
import matplotlib.pyplot as plt
import gzip
import pickle

# argv = path-to-trainset-language0 path-to-trainset-language1 path-to-trainset-language2
#        path-to-devset-data path-to-devset-labels path-to-serialize-best-model path-to-serialize-baseline


class LanguageClassifier:
    def __init__(self):
        self.language_trained_models = []
        self.prior_probabilities = []
        self.n = 0
        self.types = []
        self.most_common_label = 0

    def most_frequent_baseline(self, language_train_files):
        print("Baseline Training")
        start = time.time()
        total = 0
        for language_train_file in language_train_files:
            train_data = open(language_train_file[1], "r", encoding="utf-8")
            tokens_list = []
            # creating a list of tokens for each sentence
            for tokens in parse_incr(train_data):
                tokens_list.append(np.array(tokens))
            len_sentences = len(tokens_list)
            self.prior_probabilities.append(len_sentences)
            total += len_sentences

        self.prior_probabilities = np.asarray(self.prior_probabilities)
        self.prior_probabilities = np.log(self.prior_probabilities / total)
        self.most_common_label = np.argmax(self.prior_probabilities)

        end = time.time()
        print("\nTime elapsed training: {} seconds".format(end - start))

    def train(self, model_config, language_train_files):
        print("Training")
        total = 0
        self.language_trained_models = []
        # training
        for language_train_file in language_train_files:
            start = time.time()

            lm = NgramLanguageModel()
            lm.n = model_config[1]
            if 'sample' in language_train_file[1]:
                lm.threshold = 0
            else:
                lm.threshold = model_config[2]

            print("{} {} for language {}".format(model_config[0], lm.n, language_train_file[0]))
            print("Getting estimates for threshold: {}".format(lm.threshold))

            if model_config[0] == 'Backoff':
                if 'sample' in language_train_file[1]:
                    lm.epsilon1 = 1
                    lm.epsilon2 = 1
                else:
                    lm.epsilon1 = model_config[3]
                    lm.epsilon2 = model_config[4]

                lm.delta1 = 0.1
                lm.delta2 = 0.5

                lm.backoff(language_train_file[1], threshold=lm.threshold)
            else:
                lm.get_estimates(language_train_file[1], threshold=lm.threshold)

            end = time.time()
            print("\nTime elapsed training: {} seconds".format(end - start))

            self.language_trained_models.append([lm, language_train_file[0]])
            self.prior_probabilities.append(lm.no_of_sentences)
            total += lm.no_of_sentences

        self.prior_probabilities = np.asarray(self.prior_probabilities)
        self.prior_probabilities = np.log(self.prior_probabilities / total)

    def process_sentence(self, token_list):
        list_of_indices = [np.where(self.types == word) for word in token_list]
        if self.n == 1:
            return list_of_indices
        else:
            list_bigrams = []
            for i in range(len(list_of_indices)-1):
                list_bigrams.append((list_of_indices[i], list_of_indices[i+1]))
            return list_bigrams

    def eval(self, tokens_list):
        print("Evaluating")
        start = time.time()
        log_likelihoods_plus_prior = []

        for i, model in enumerate(self.language_trained_models):
            lm_eval = model[0]
            self.n = lm_eval.n
            self.types = lm_eval.types
            if lm_eval.n == 1:
                tokens_list_language = np.array(list(map(lm_eval.add_UNK, tokens_list)))
            else:
                tokens_list_language = np.array(list(map(lm_eval.pad_both_ends_add_UNK, tokens_list)))

            processed_tokens_list = list(map(self.process_sentence, tokens_list_language))

            log_likelihoods = []
            for token_list in processed_tokens_list:
                log_likelihoods.append(lm_eval.calc_eval_prob(token_list))

            log_likelihoods = np.asarray(log_likelihoods)

            log_likelihoods = np.negative(np.add(log_likelihoods, self.prior_probabilities[i]))
            log_likelihoods_plus_prior.append(log_likelihoods)


        log_probabilities = list(zip(log_likelihoods_plus_prior[0], log_likelihoods_plus_prior[1],
                                     log_likelihoods_plus_prior[2]))
        eval_labels = [np.argmax(log_prob) for log_prob in log_probabilities]

        end = time.time()
        print("\nTime elapsed evaluating: {} seconds".format(end - start))

        return eval_labels


if __name__ == '__main__':
    print("Language Classification")
    language_train_files = [(0, sys.argv[1]), (1, sys.argv[2]), (2, sys.argv[3])]
    language_eval_file = sys.argv[4]
    language_eval_labels = sys.argv[5]
    # 0-ewt 1-gum 2-hin
    labels = np.array(['ewt', 'gum', 'hin'])
    if 'sample' in language_train_files[0][1]:
        model_configs = [('Backoff', 1, 0, 1, 1), ('Backoff', 2, 0, 1, 1), ('Backoff', 2, 0, 1, 1), ('Backoff', 2, 0, 1, 1)]
    else:
        model_configs = [('Backoff', 1, 15, 20, 20), ('Backoff', 1, 30, 40, 40),
                         ('Backoff', 2, 15, 20, 20), ('Backoff', 2, 20, 40, 40)]
    lcs = []
    f1_scores = []

    dev_data = open(language_eval_file, "r", encoding="utf-8")
    tokens_list = []
    # creating a list of tokens for each sentence
    for tokens in parse_incr(dev_data):
        tokens_list.append(np.array(tokens))
    tokens_list = np.array(tokens_list)
    # extracting only the second column of each entry and making a list of all the tokens
    tokens_list = [[t['form'] for t in token] for token in tokens_list]

    label_data = open(language_eval_labels, "r", encoding="utf-8")
    gold_labels = label_data.readlines()
    gold_labels = np.asarray([label.strip() for label in gold_labels])
    # print(gold_labels)
    gold_labels = [np.where(labels == label)[0][0] for label in gold_labels]

    # most common baseline
    baseline = LanguageClassifier()
    baseline.most_frequent_baseline(language_train_files)
    baseline_eval_labels = np.repeat(baseline.most_common_label, len(gold_labels))
    if 'sample' in language_train_files[0][1]:
        print("Pred labels: {}".format(baseline_eval_labels))
        print("True labels: {}".format(gold_labels))
    f1_score = skm.f1_score(gold_labels, baseline_eval_labels, average='macro')
    print("F1 score: {}".format(f1_score))
    f1_scores.append(f1_score)
    lcs.append([('Baseline', ''), baseline])

    # model configs
    for model_config in model_configs:
        lc = LanguageClassifier()
        lc.train(model_config, language_train_files)
        eval_labels = lc.eval(tokens_list)
        if 'sample' in language_train_files[0][1]:
            print("Pred labels: {}".format(eval_labels))
            print("True labels: {}".format(gold_labels))
        f1_score = skm.f1_score(gold_labels, eval_labels, average='macro')
        print("F1 score: {}".format(f1_score))
        f1_scores.append(f1_score)
        lcs.append([model_config, lc])

        print()
        print()

    model_configs.insert(0, ('Baseline', '', '', '', ''))
    print("F1 scores: {}".format(list(zip(model_configs, f1_scores))))

    best_model_index = np.argmax(f1_scores)
    best_model = lcs[best_model_index][1]

    print("Best model config: {}".format(model_configs[best_model_index]))
    with gzip.open(sys.argv[6], 'wb') as fp:
        pickle.dump(best_model, fp)

    with gzip.open(sys.argv[7], 'wb') as fp:
        pickle.dump(baseline, fp)

    x = [str(config[0])+' '+str(config[1])+':'+str(config[2]) for config in model_configs]
    plt.plot(x, f1_scores, 'b-o')
    plt.xlabel('Models')
    plt.ylabel('F1 scores')
    plt.show()
