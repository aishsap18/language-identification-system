import pickle
import gzip
import time
import sys
from model import NgramLanguageModel

# argv = path-to-serialized-model path-to-dev-set

if __name__ == '__main__':

    lm = NgramLanguageModel()
    fp = gzip.open(sys.argv[1], 'rb')
    lm = pickle.load(fp)

    start = time.time()
    print("Starting dev for threshold: {}".format(lm.threshold))
    dev_perplexity = lm.get_dev_perplexity(sys.argv[2])
    print("Dev perplexity: {}".format(dev_perplexity))
    end = time.time()
    print("\n\nTime elapsed evaluating: {} seconds".format(end - start))

