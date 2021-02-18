import pickle
import gzip
import time
import sys
from model import NgramLanguageModel
import matplotlib.pyplot as plt
import numpy as np

# argv = path-to-serialized-model-mle path-to-serialized-model-laplace path-to-serialized-model-backoff path-to-test-set

if __name__ == '__main__':

    models = [('MLE', sys.argv[1]), ('Laplace', sys.argv[2]), ('Backoff', sys.argv[3])]
    test_ppl = []

    for model in models:
        lm = NgramLanguageModel()
        fp = gzip.open(model[1], 'rb')
        lm = pickle.load(fp)

        start = time.time()
        print("{}: {}".format(model[0], lm.n))
        print("Starting dev for threshold: {}".format(lm.threshold))
        dev_perplexity = lm.get_dev_perplexity(sys.argv[4])
        print("Test perplexity: {}".format(dev_perplexity))
        end = time.time()
        print("\n\nTime elapsed evaluating: {} seconds".format(end - start))

        # test_ppl.append([model[0]+': '+str(lm.n), lm.threshold, dev_perplexity])
        test_ppl.append(dev_perplexity)

    fig1 = plt.figure()
    # x = ['MLE: '+test_ppl[0][0], 'Laplace: '+test_ppl[1][0], 'Backoff: '+test_ppl[2][0]]
    models = np.asarray(models)
    plt.plot(models[:, 0], test_ppl, 'b-o')
    plt.xlabel('Threshold')
    plt.ylabel('Dev perplexity')

    plt.tight_layout()
    plt.show()