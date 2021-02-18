from model import NgramLanguageModel
import pickle
import gzip
import time
import sys
import matplotlib as mt
mt.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# argv = path_to_train_file path_to_tune_file path-to-save-best-mle path-to-save-best-laplace path-to-save-best-backoff
fig = plt.figure(figsize=(11, 4))
i = 131


def show_plot(dev_ppl_list):
    global i

    ax = fig.add_subplot(i)
    ax.title.set_text(dev_ppl_list[0][0])
    if len(dev_ppl_list) < 6:
        plt.plot(dev_ppl_list[:, 2], dev_ppl_list[:, 3], 'r-o')
        plt.legend(['unigram'])
    else:
        plt.plot(dev_ppl_list[:, 2][:3], dev_ppl_list[:, 3][:3], 'b-o',
                 dev_ppl_list[:, 2][3:], dev_ppl_list[:, 3][3:], 'r-o')
        plt.legend(['unigram', 'bigram'])
    plt.xlabel('Threshold')
    plt.ylabel('Dev perplexity')
    i += 1


if __name__ == '__main__':

    dev_ppl_mle = []
    dev_ppl_laplace = []
    dev_ppl_backoff = []

    if "sample" in sys.argv[1]:
        pairs = [(1, 0), (1, 1), (1, 1), (2, 0), (2, 1), (2, 1)]
    else:
        pairs = [(1, 5), (1, 10), (1, 15), (2, 5), (2, 10), (2, 15)]

    # mle configs
    for pair in pairs:
        start = time.time()

        lm = NgramLanguageModel()
        lm.n = pair[0]
        lm.threshold = pair[1]

        print("MLE: {}".format(lm.n))
        print("Getting estimates for threshold: {}".format(lm.threshold))
        lm.get_estimates(sys.argv[1], threshold=lm.threshold)

        end = time.time()
        print("\nTime elapsed training: {} seconds".format(end - start))

        start = time.time()
        print("Starting dev")
        dev_perplexity = lm.get_dev_perplexity(sys.argv[2])
        print("Dev perplexity: {}".format(dev_perplexity))
        end = time.time()
        print("\nTime elapsed evaluating: {} seconds".format(end - start))

        if dev_perplexity != float('inf'):
            dev_ppl_mle.append(['MLE', lm.n, lm.threshold, dev_perplexity, lm])

    # laplace configs
    for pair in pairs:
        start = time.time()

        lm = NgramLanguageModel()
        lm.n = pair[0]
        lm.threshold = pair[1]
        lm.lambda_val = 1

        print("Laplace: {}".format(lm.n))
        print("Getting estimates for threshold: {}".format(lm.threshold))
        print("Lambda value: {}".format(lm.lambda_val))
        lm.get_estimates(sys.argv[1], threshold=lm.threshold)

        end = time.time()
        print("\nTime elapsed training: {} seconds".format(end - start))

        start = time.time()
        print("Starting dev")
        dev_perplexity = lm.get_dev_perplexity(sys.argv[2])
        print("Dev perplexity: {}".format(dev_perplexity))
        end = time.time()
        print("\nTime elapsed evaluating: {} seconds".format(end - start))

        dev_ppl_laplace.append(['Laplace', lm.n, lm.threshold, dev_perplexity, lm])

    # backoff configs
    for pair in pairs:
        start = time.time()

        lm = NgramLanguageModel()
        lm.n = pair[0]
        lm.threshold = pair[1]
        if "sample" in sys.argv[1]:
            lm.epsilon1 = 1
            lm.epsilon2 = 1
        else:
            lm.epsilon1 = 20
            lm.epsilon2 = 20
        lm.delta1 = 0.1
        lm.delta2 = 0.5

        print("Backoff: {}".format(lm.n))
        print("Getting estimates for threshold: {}".format(lm.threshold))
        print("Epsilon 1: {}, Epsilon 2: {}, Delta 1: {}, Delta 2: {}"
              .format(lm.epsilon1, lm.epsilon2, lm.delta1, lm.delta2))
        lm.backoff(sys.argv[1], threshold=lm.threshold)

        end = time.time()
        print("\nTime elapsed training: {} seconds".format(end - start))

        start = time.time()
        print("Starting dev")
        dev_perplexity = lm.get_dev_perplexity(sys.argv[2])
        print("Dev perplexity: {}".format(dev_perplexity))
        end = time.time()
        print("\nTime elapsed evaluating: {} seconds".format(end - start))

        dev_ppl_backoff.append(['Backoff', lm.n, lm.threshold, dev_perplexity, lm])

    dev_ppl_mle = np.asarray(dev_ppl_mle)
    dev_ppl_laplace = np.asarray(dev_ppl_laplace)
    dev_ppl_backoff = np.asarray(dev_ppl_backoff)

    best_lm_mle = sorted(dev_ppl_mle, key=lambda x: x[3])[0]
    print("Lowest dev perplexity in MLE: {}".format(best_lm_mle[3]))
    fp = gzip.open(sys.argv[3], 'wb')
    pickle.dump(best_lm_mle[4], fp)

    best_lm_laplace = sorted(dev_ppl_laplace, key=lambda x: x[3])[0]
    print("Lowest dev perplexity in Laplace: {}".format(best_lm_laplace[3]))
    fp = gzip.open(sys.argv[4], 'wb')
    pickle.dump(best_lm_laplace[4], fp)

    best_lm_backoff = sorted(dev_ppl_backoff, key=lambda x: x[3])[0]
    print("Lowest dev perplexity in Backoff: {}".format(best_lm_backoff[3]))
    fp = gzip.open(sys.argv[5], 'wb')
    pickle.dump(best_lm_backoff[4], fp)

    show_plot(dev_ppl_mle)
    show_plot(dev_ppl_laplace)
    show_plot(dev_ppl_backoff)

    fig1 = plt.figure(figsize=(9, 3))
    ax = fig1.add_subplot(121)
    ax.title.set_text('Unigram')
    plt.plot(dev_ppl_laplace[:, 2][:3], dev_ppl_laplace[:, 3][:3], 'r-o',
             dev_ppl_backoff[:, 2][:3], dev_ppl_backoff[:, 3][:3], 'g-o',
             dev_ppl_mle[:, 2], dev_ppl_mle[:, 3], 'b-o', alpha=0.5)
    plt.legend(['laplace', 'backoff', 'mle'])
    plt.xlabel('Threshold')
    plt.ylabel('Dev perplexity')

    ax = fig1.add_subplot(122)
    ax.title.set_text('Bigram')
    plt.plot(dev_ppl_laplace[:, 2][3:], dev_ppl_laplace[:, 3][3:], 'r-o',
             dev_ppl_backoff[:, 2][3:], dev_ppl_backoff[:, 3][3:], 'g-o',
             dev_ppl_mle[:, 2][3:], dev_ppl_mle[:, 3][3:], 'b-o', alpha=0.5)
    plt.legend(['laplace', 'backoff', 'mle'])
    plt.xlabel('Threshold')
    plt.ylabel('Dev perplexity')

    plt.tight_layout()
    plt.show()
