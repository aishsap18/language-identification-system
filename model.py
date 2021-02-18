import sys
from conllu import parse_incr
import numpy as np
import time
import pickle
import gzip


class NgramLanguageModel:
    threshold = 0
    n = 0
    lambda_val = 0
    types = []
    total_no_types_uni = 0
    total_no_types_bi = 0
    no_of_sentences = 0
    unigram_counts = []
    unigram_prob = []
    bigram_counts = []
    total_no_tokens = 0
    bigram_matrix = []
    bigram_matrix_dev = []
    bigram_indices = []
    bigram_prob = []
    z_values_per_row = []

    def pad_both_ends_add_UNK(self, token_list):
        token_list = np.array([token.replace(token, '<UNK>') if token not in self.types else token
                               for token in token_list],
                              dtype='object')
        token_list = np.insert(token_list, 0, '<BOS>', axis=0)
        token_list = np.append(token_list, '<EOS>')
        return token_list

    def add_UNK(self, token_list):
        return np.array([token.replace(token, '<UNK>')
                         if token not in self.types
                         else token
                         for token in token_list],
                        dtype='object')

    def get_estimates(self, train_file_path, threshold=threshold, calc_prob=True):
        # loading data from en_ewt-ud-train.conllu file
        train_data = open(train_file_path, "r", encoding="utf-8")
        tokens_list = []

        # creating a list of tokens for each sentence
        for tokens in parse_incr(train_data):
            tokens_list.append(np.array(tokens))

        tokens_list = np.array(tokens_list)

        # extracting only the second column of each entry and making a list of all the tokens
        tokens_list = [np.array([t['form'] for t in token], dtype='object') for token in tokens_list]
        self.no_of_sentences = len(tokens_list)

        # combining all the tokens into a single list
        combined_tokens_train = np.concatenate(tokens_list)

        # getting all the distinct tokens to form the types list
        # along with their count of occurrence in the full training file
        types_train, counts = np.unique(combined_tokens_train, return_counts=True)

        # put hyperparameter tuning logic here
        threshold = int(threshold)

        if threshold >= 1:
            types_train = list(zip(types_train, counts))

            # removing the types with values less than hyper parameter threshold
            types_train_tuned = [(self.types, count) for self.types, count in types_train if count > threshold]
            # print(types_train_tuned)
            types_train_tuned, counts_tuned = list(zip(*types_train_tuned))
            types_train_tuned = list(types_train_tuned)

            if self.n == 1:
                counts_tuned = list(counts_tuned)

                count_of_UNKs = sum(counts) - sum(counts_tuned)
                counts_tuned = np.insert(counts_tuned, 0, count_of_UNKs)

                self.unigram_counts = np.asarray(counts_tuned)

            # types_train_tuned = np.insert(types_train_tuned, 0, '<UNK>')
            self.types = np.asarray(types_train_tuned, dtype='object')
            self.types = np.insert(self.types, 0, '<UNK>')

        else:
            self.types = np.asarray(types_train, dtype='object')
            self.unigram_counts = counts

            self.types = np.insert(self.types, 0, '<UNK>')
            self.unigram_counts = np.insert(self.unigram_counts, 0, 0)

        if self.n == 1:
            if self.lambda_val > 0:
                self.unigram_counts = self.unigram_counts + self.lambda_val

            self.total_no_tokens = np.sum(self.unigram_counts)
            self.total_no_types_uni = len(self.types)
            if calc_prob:
                self.unigram_prob = self.unigram_counts / self.total_no_tokens

        if self.n == 2:
            print("Padding start and end and replacing UNK")
            # pad start and end of sentence with tokens <BOS> and <EOS>
            # along with replace <UNK> for unknown words
            tokens_list = list(map(self.pad_both_ends_add_UNK, tokens_list))

            self.types = np.append(self.types, ['<BOS>', '<EOS>'])

            self.total_no_types_bi = len(self.types)

            self.bigram_matrix = np.zeros((len(self.types), len(self.types)), np.float64)

            print()
            print("Computing bigram matrix.")

            # for each window populate the bigram matrix
            list(map(self.process_window, tokens_list))

            # create a list of all non zero counts of word pairs
            self.bigram_counts = self.bigram_matrix[np.nonzero(self.bigram_matrix)]

            if self.lambda_val > 0:
                self.bigram_counts = self.bigram_counts + self.lambda_val

            # preserve the indices for computation
            self.bigram_indices = np.transpose(np.nonzero(self.bigram_matrix))

            if calc_prob:
                print("Computing probabilities.")
                self.z_values_per_row = np.full(len(self.types), (self.total_no_types_bi-1), dtype=np.float64)
                self.bigram_prob = list(map(self.calculate_norm_bigram_prob, range(len(self.bigram_indices))))

            self.bigram_indices = self.bigram_indices.tolist()

    def calculate_norm_bigram_prob(self, bi_index):
        x = self.bigram_indices[bi_index][0]
        z_indexes = np.where(self.bigram_indices[:, 0] == x)

        if self.lambda_val > 0:
            # to marginalize y add all the lambda values which are not found in types
            # -1 to not consider <BOS>
            no_of_zero_values = (self.total_no_types_bi - 1) - len(z_indexes[0])

            z = sum(self.bigram_counts[z_indexes]) + (no_of_zero_values * self.lambda_val)
            # storing the z value for each row to calc the probability of 0 values
            self.z_values_per_row[x] = z
        else:
            z = sum(self.bigram_counts[z_indexes])

        return self.bigram_counts[bi_index] / z

    def process_window(self, window, func='train'):
        ''' Function to populate bigram matrix if func_call is from get_estimates for training
            else it creates list of dev bigram indexes based on list_of_indices for evaluating '''
        # global bigram_matrix, types, bigram_matrix_dev

        # get the list of indices for each word in current window
        list_of_indices = [np.where(self.types == word) for word in window]

        for i in range(len(list_of_indices) - 1):
            if func == 'train':
                # add the count
                self.bigram_matrix[list_of_indices[i][0][0], list_of_indices[i + 1][0][0]] += 1
            else:
                self.bigram_matrix_dev[list_of_indices[i][0][0], list_of_indices[i + 1][0][0]] += 1

    def get_dev_perplexity(self, dev_file_path):
        # loading data from en_ewt-ud-train.conllu file
        dev_data = open(dev_file_path, "r", encoding="utf-8")
        tokens_list = []

        # creating a list of tokens for each sentence
        for tokens in parse_incr(dev_data):
            tokens_list.append(np.array(tokens))

        tokens_list = np.array(tokens_list)

        # extracting only the second column of each entry and making a list of all the tokens
        tokens_list = [[t['form'] for t in token] for token in tokens_list]
        # print(tokens_list)

        if self.n == 2:
            print("Padding start end and replacing UNK in dev data.")
            tokens_list = list(map(self.pad_both_ends_add_UNK, tokens_list))

            print("Processing dev windows.")
            self.bigram_matrix_dev = np.zeros((len(self.types), len(self.types)), np.float64)
            list(map(self.process_window, tokens_list, np.repeat('dev', len(tokens_list))))
            dev_bigram_counts = self.bigram_matrix_dev[np.nonzero(self.bigram_matrix_dev)]
            dev_bigram_indices = np.transpose(np.nonzero(self.bigram_matrix_dev))

            print("Calculating dev perplexity.")
            dev_bigram_indices = dev_bigram_indices.tolist()
            dev_perplexity_list = list(map(self.calc_dev_ppl, dev_bigram_indices, dev_bigram_counts))

            dev_perplexity_list = np.asarray(dev_perplexity_list)
            dev_perplexity = np.exp(-(sum(dev_perplexity_list) / sum(dev_bigram_counts)))
        else:
            print("Calculating dev perplexity.")
            tokens_list = np.array(list(map(self.add_UNK, tokens_list)))
            combined_tokens_train = np.concatenate(tokens_list)
            dev_types, dev_counts = np.unique(combined_tokens_train, return_counts=True)

            ind0, ind_types, ind1 = np.intersect1d(self.types, dev_types, return_indices=True)
            ind0 = []
            ind1 = []
            dev_perplexity_list = list(map(self.calc_dev_ppl, ind_types, dev_counts))

            dev_perplexity_list = np.asarray(dev_perplexity_list)
            dev_perplexity = np.exp(-(sum(dev_perplexity_list) / sum(dev_counts)))

        return dev_perplexity

    def calc_dev_ppl(self, dev_token, dev_count):
        if self.n == 1:
            p_x = self.unigram_prob[dev_token]
            if p_x == 0 and self.backoff_flag == 1:
                p_x = self.beta / self.total_no_types_uni

            np.seterr(all='ignore')
            dev_ppl = dev_count * np.log(p_x)
        else:
            if self.backoff_flag == 1:
                p_x_y = self.bigram_prob[dev_token[0]][dev_token[1]]
            else:
                if dev_token in self.bigram_indices:
                    index = self.bigram_indices.index(dev_token)
                    p_x_y = self.bigram_prob[index]
                else:
                    if self.lambda_val == 0:
                        p_x_y = 0
                    else:
                        z = self.z_values_per_row[dev_token[0]]
                        p_x_y = self.lambda_val / z

            np.seterr(all='ignore')
            dev_ppl = dev_count * np.log(p_x_y)

        return dev_ppl

    # for language classification
    def calc_eval_prob(self, token_list):
        if self.n == 1:
            log_prob = 0
            for token in token_list:
                p_x = self.unigram_prob[token]
                if p_x == 0 and self.backoff_flag == 1:
                    p_x = self.beta / self.total_no_types_uni

                log_prob += np.log(p_x)
        else:
            log_prob = 0
            for token in token_list:
                if self.backoff_flag == 1:
                    p_x_y = self.bigram_prob[token[0][0][0]][token[1][0][0]]
                else:
                    token = [token[0][0][0], token[1][0][0]]
                    if token in self.bigram_indices:
                        index = self.bigram_indices.index(token)
                        p_x_y = self.bigram_prob[index]
                    else:
                        if self.lambda_val == 0:
                            p_x_y = 0
                        else:
                            z = self.z_values_per_row[token[0]]
                            p_x_y = self.lambda_val / z

                log_prob += np.log(p_x_y)
        return log_prob

    epsilon1 = 1
    epsilon2 = 1
    delta1 = 0.1
    delta2 = 0.5
    beta = 0
    backoff_flag = 0
    alphas = []
    initial_backed_uni_prob = []

    def backoff(self, train_file_path, threshold):
        self.backoff_flag = 1

        flag = 0
        if self.n == 2:
            self.n = 1
            flag = 1

        self.get_estimates(train_file_path, threshold, calc_prob=False)
        if 'sample' in train_file_path:
            print("Types: {}".format(len(self.types)))
            print(self.types)
            print("Counts: {}".format(self.unigram_counts))
            print("Total number of tokens: {}".format(self.total_no_tokens))

        no_of_epsilon_counts = len(self.unigram_counts[self.unigram_counts < self.epsilon1])
        if no_of_epsilon_counts > 0:
            self.unigram_prob = np.zeros(self.total_no_types_uni)
            unigrams = list(zip(range(len(self.unigram_counts)), self.unigram_counts))
            list(map(self.calc_greater_than_epsilon_prob_uni, unigrams))
        else:
            self.unigram_prob = self.unigram_counts / self.total_no_tokens

        # probability of each less than epsilon item
        no_of_non_zeros = np.count_nonzero(self.unigram_prob)
        no_of_zeros = self.total_no_types_uni - no_of_non_zeros

        if no_of_zeros > 0:
            self.beta = (1 - sum(self.unigram_prob)) * self.total_no_types_uni / no_of_zeros

        if 'sample' in train_file_path:
            print("Number of items with count more than epsilon: {}".format(no_of_non_zeros))
            print("Number of items with count less than epsilon: {}".format(no_of_zeros))
            if no_of_zeros > 0:
                prob_of_zeros = no_of_zeros * (self.beta / self.total_no_types_uni)
                total_prob = sum(self.unigram_prob) + prob_of_zeros
            else:
                total_prob = sum(self.unigram_prob)

            print("Total: {}".format(total_prob))

            print("Unigram Prob: {}".format(self.unigram_prob))
            print("Prob of less than epsilon items: {}".format(self.beta / self.total_no_types_uni))
            print()

        if flag == 1:
            self.n = 2

        if self.n == 2:
            self.get_estimates(train_file_path, threshold, calc_prob=False)
            if 'sample' in train_file_path:
                print("Types: {}".format(len(self.types)))
                print(self.types)
                print("Bigram indices: {}".format(self.bigram_indices))
                print("Counts: {}".format(self.bigram_counts))

            self.bigram_prob = np.zeros((len(self.types), len(self.types)), np.float64)
            list(map(self.calc_backoff_prob_bi, range(len(self.bigram_matrix))))

            if 'sample' in train_file_path:
                for i, row in enumerate(self.bigram_prob):
                    print("Total for row {}: {}".format(i, sum(row)))

    def calc_greater_than_epsilon_prob_uni(self, unigram):
        index = unigram[0]
        count = unigram[1]
        if count > self.epsilon1:
            prob = (count - self.delta1) / self.total_no_tokens
            self.unigram_prob[index] = prob

    def calc_backoff_prob_bi(self, bi_index):
        z = sum(self.bigram_matrix[bi_index])
        non_eplison_indexes = []
        non_epsilon_prob = []
        epsilon_prob = []
        for i, item in enumerate(self.bigram_matrix[bi_index]):
            # prob of <BOS> 0 as we do not have to observe them
            if i == self.total_no_types_bi - 2:
                prob = 0
            elif item > self.epsilon2:
                prob = (item - self.delta2) / z
                non_epsilon_prob.append(prob)
            else:
                # prob of <EOS> = c(<EOS>) / (total_no_tokens + no_of_sentences)
                if i == self.total_no_types_bi - 1:
                    prob = self.no_of_sentences / (self.total_no_tokens + self.no_of_sentences)
                else:
                    prob = self.unigram_prob[i]
                    if prob == 0:
                        prob = self.beta / self.total_no_types_uni

                epsilon_prob.append(prob)
                non_eplison_indexes.append(i)

            self.bigram_prob[bi_index][i] = prob

        alpha = (1 - sum(non_epsilon_prob)) / sum(epsilon_prob)
        self.bigram_prob[bi_index][non_eplison_indexes] = self.bigram_prob[bi_index][non_eplison_indexes] * alpha


if __name__ == '__main__':

    start = time.time()

    threshold_value = 0
    model_name = sys.argv[1]

    if model_name == 'mle':
        # argv = model_name n path_to_train_file path_to_tune_file path_to_serialized_model [threshold]
        print("MLE")
        if len(sys.argv) >= 7:
            threshold_value = float(sys.argv[6])

    elif model_name == 'laplace':
        # argv = model_name n path_to_train_file path_to_tune_file path_to_serialized_model [lambda threshold]
        print("Laplace")
        lambda_val = 1
        if len(sys.argv) >= 7:
            lambda_val = float(sys.argv[6])
        if len(sys.argv) >= 8:
            threshold_value = float(sys.argv[7])
    else:
        # argv = model_name n path_to_train_file path_to_tune_file path_to_serialized_model [epsilon1 epsilon2 delta1 delta2 threshold]
        print("Backoff")
        if len(sys.argv) >= 11:
            threshold_value = float(sys.argv[10])

    lm = NgramLanguageModel()
    lm.threshold = threshold_value
    lm.n = int(sys.argv[2])

    if sys.argv[1] == 'laplace':
        lm.lambda_val = lambda_val
        print("Lambda value: {}".format(lambda_val))

    if sys.argv[1] == 'backoff':
        if len(sys.argv) >= 7:
            lm.epsilon1 = float(sys.argv[6])
        if len(sys.argv) >= 8:
            lm.epsilon2 = float(sys.argv[7])
        if len(sys.argv) >= 9:
            lm.delta1 = float(sys.argv[8])
        if len(sys.argv) >= 10:
            lm.delta2 = float(sys.argv[9])

        lm.backoff(sys.argv[3], threshold=lm.threshold)
    else:
        print("Getting estimates for threshold: {}".format(lm.threshold))
        lm.get_estimates(sys.argv[3], threshold=lm.threshold)
        if 'sample' in sys.argv[3] and lm.n == 1:
            print("Types: {}".format(lm.total_no_types_uni))
            print(lm.types)
            print("Counts: {}".format(lm.unigram_counts))
            print("Total number of tokens: {}".format(lm.total_no_tokens))
            print("Probabilities: {}".format(lm.unigram_prob))
            print("Total of all probs: {}".format(sum(lm.unigram_prob)))
        if 'sample' in sys.argv[3] and lm.n == 2:
            print("Types: {}".format(lm.total_no_types_bi))
            print(lm.types)
            print("Bigram indices: {}".format(lm.bigram_indices))
            print("Counts: {}".format(lm.bigram_counts))
            print("Probabilities: {}".format(lm.bigram_prob))
            # each row should result in total prob 1 as we are marginalizing over y
            # taking each row values present in bigram_probs
            print("Total of probabilities per row: ", end='\t')
            sep_lists = list()
            for sep in range(lm.total_no_types_bi - 1):
                sep_lists.append(list())
            for j, item in enumerate(lm.bigram_indices):
                sep_lists[item[0]].append(j)

            for j, sep in enumerate(sep_lists):
                no_of_zeros = (lm.total_no_types_bi - 1) - len(sep)
                total = (lm.lambda_val / lm.z_values_per_row[j]) * no_of_zeros
                for item in sep:
                    total += lm.bigram_prob[item]
                print(total, end='\t')

    end = time.time()
    print("\nTime elapsed training: {} seconds".format(end - start))

    start = time.time()
    print("Starting dev")
    dev_perplexity = lm.get_dev_perplexity(sys.argv[4])
    print("Dev perplexity: {}".format(dev_perplexity))
    end = time.time()
    print("\nTime elapsed evaluating: {} seconds".format(end - start))

    fp = gzip.open(sys.argv[5], 'wb')
    pickle.dump(lm, fp)
