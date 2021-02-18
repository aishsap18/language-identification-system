import sys
from conllu import parse_incr
import numpy as np

# argv = path-to-dataset-language1 path-to-dataset-language2 path-to-dataset-language2
#        path-to-save-merged-data-file path-to-save-merged-labels-file

if __name__ == '__main__':

    labels = ['ewt', 'gum', 'hin']
    language_file_paths = [sys.argv[1], sys.argv[2], sys.argv[3]]

    languages = list(zip(language_file_paths, labels))
    tokens_list = []

    print("Processing files ...")
    for language in languages:
        # loading data from en_ewt-ud-train.conllu file
        data = open(language[0], "r", encoding="utf-8")

        # creating a list of tokens for each sentence
        for tokens in parse_incr(data):
            tokens_list.append([tokens.serialize(), language[1]])

    tokens_list = np.asarray(tokens_list)

    np.random.shuffle(tokens_list)
    # print(tokens_list)

    labels_list = tokens_list[:, 1]
    # print(labels_list)
    tokens_list = tokens_list[:, 0]
    # print(tokens_list)

    print("Writing files ...")
    with open(sys.argv[4], 'w', encoding="utf-8") as fp:
        [fp.write("%s" % item) for item in tokens_list]

    with open(sys.argv[5], 'w', encoding="utf-8") as fp:
        [fp.write("%s\n" % item) for item in labels_list]
