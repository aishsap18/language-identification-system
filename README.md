# Language Identification System

Given a sentence, identify the likely language that the sentence is written in among English-EWT, English-GUM and Hindi based on the language data in [Universal Dependencies](https://universaldependencies.org/) dataset. 

Note - The language and classification models are implemented from scratch in Python 3.7.3. The bash files are used only to make the module easily accessible.


### Installation

- Create virtual env:
	``` $ virtualenv lang_idf_env ```

- Activate the env:
	``` $ source lang_idf_env/env/activate ```

- To install all the packages in requirements.txt file:
	``` (lang_idf_env)$ pip install -r requirements.txt ```

## Steps to execute

#### Language Modeling 

I. MLE & Laplace

- MLE Parameters to `ngram_lm_train.bash` - model_name, n, path_to_train_file, path_to_tune_file, path_to_serialized_model, (optional) [threshold]

    1. Unigram 
	``` (lang_idf_env)$ bash ngram_lm_train.bash mle 1 data/UD_English-EWT/en_ewt-ud-train.conllu data/UD_English-EWT/en_ewt-ud-dev.conllu save_models/mle1.pickle 0 ```

    2. Bigram
	``` (lang_idf_env)$ bash ngram_lm_train.bash mle 2 data/UD_English-EWT/en_ewt-ud-train.conllu data/UD_English-EWT/en_ewt-ud-dev.conllu save_models/mle2.pickle 10 ```

- Laplace Parameters to `ngram_lm_train.bash` - model_name, n, path_to_train_file, path_to_tune_file, path_to_serialized_model, (optional) [lambda, threshold]
		
    1. Unigram
	``` (lang_idf_env)$ bash ngram_lm_train.bash laplace 1 data/English-EWT/en_ewt-ud-train.conllu data/English-EWT/en_ewt-ud-dev.conllu save_models/laplace1.pickle 1 0 ``` 

    2. Bigram
	``` (lang_idf_env)$ bash ngram_lm_train.bash laplace 2 data/English-EWT/en_ewt-ud-train.conllu data/English-EWT/en_ewt-ud-dev.conllu save_models/laplace2.pickle 1 10 ``` 


II. Serialize/Deserialize

- Parameters to `ngram_lm_eval.bash` - path-to-serialized-model, path-to-dev-set

    1. MLE Unigram
	``` (lang_idf_env)$ bash ngram_lm_eval.bash save_models/mle1.pickle data/English-EWT/en_ewt-ud-dev.conllu ``` 

    2. MLE Bigram
	``` (lang_idf_env)$ bash ngram_lm_eval.bash save_models/mle2.pickle data/English-EWT/en_ewt-ud-dev.conllu ``` 

    3. Laplace Unigram
	``` (lang_idf_env)$ bash ngram_lm_eval.bash save_models/laplace1.pickle data/English-EWT/en_ewt-ud-dev.conllu ``` 

    4. Laplace Bigram
	``` (lang_idf_env)$ bash ngram_lm_eval.bash save_models/laplace2.pickle data/English-EWT/en_ewt-ud-dev.conllu ``` 


III. Backoff 

- Train Parameters to `ngram_lm_train.bash` - model_name, n, path_to_train_file, path_to_tune_file, path_to_serialized_model, (optional) [epsilon1, epsilon2, delta1, delta2, threshold]

    1. Backoff Unigram
	``` (lang_idf_env)$ bash ngram_lm_train.bash backoff 1 data/English-EWT/en_ewt-ud-train.conllu data/English-EWT/en_ewt-ud-dev.conllu save_models/backoff1.pickle 1 1 0.1 0.5 0 ``` 

    2. Backoff Bigram
	``` (lang_idf_env)$ bash ngram_lm_train.bash backoff 2 data/English-EWT/en_ewt-ud-train.conllu data/English-EWT/en_ewt-ud-dev.conllu save_models/backoff2.pickle 15 15 0.1 0.5 10``` 

- Eval Parameters to `ngram_lm_eval.bash` - path-to-serialized-model, path-to-dev-set

    1. Backoff Unigram
	``` (lang_idf_env)$ bash ngram_lm_eval.bash save_models/backoff1.pickle data/English-EWT/en_ewt-ud-dev.conllu``` 

    2. Backoff Bigram
	``` (lang_idf_env)$ bash ngram_lm_eval.bash save_models/backoff2.pickle data/English-EWT/en_ewt-ud-dev.conllu``` 


IV. Configurations

- Parameters to `ngram_lm_configs.bash` - path_to_train_file, path_to_tune_file, path-to-save-best-mle, path-to-save-best-laplace, path-to-save-best-backoff

    Run the config file:
	``` (lang_idf_env)$ bash ngram_lm_configs.bash data/English-EWT/en_ewt-ud-train.conllu data/English-EWT/en_ewt-ud-dev.conllu save_best/best_mle.pickle save_best/best_laplace.pickle save_best/best_backoff.pickle``` 


#### Language Classification

V. Merge datasets parameters to `merge_sets.bash` - path-to-dataset-language1, path-to-dataset-language2, path-to-dataset-language3, path-to-save-merged-data-file, path-to-save-merged-labels-file

- Merge dev sets - 
	``` (lang_idf_env)$ bash merge_sets.bash data/English-EWT/en_ewt-ud-dev.conllu data/English-GUM/en_gum-ud-dev.conllu data/Hindi-HDTB/hi_hdtb-ud-dev.conllu merged_sets/dev_set.conllu merged_sets/dev_labels.txt``` 

- Merge test sets - 
	``` (lang_idf_env)$ bash merge_sets.bash data/English-EWT/en_ewt-ud-test.conllu data/English-GUM/en_gum-ud-test.conllu data/Hindi-HDTB/hi_hdtb-ud-test.conllu merged_sets/test_set.conllu merged_sets/test_labels.txt``` 

VI. Train Parameters to `ngram_lc_train.bash` - path-to-trainset-language1, path-to-trainset-language2, path-to-trainset-language3, path-to-devset-data, path-to-devset-labels, path-to-serialize-best-model, path-to-serialize-baseline 

- Train Models - 
	``` (lang_idf_env)$ bash ngram_lc_train.bash data/English-EWT/en_ewt-ud-train.conllu data/English-GUM/en_gum-ud-train.conllu data/Hindi-HDTB/hi_hdtb-ud-train.conllu merged_sets/dev_set.conllu merged_sets/dev_labels.txt save_best/best_language_classifier.pickle save_best/baseline_classifier.pickle``` 

VII. Test eval Parameters to `ngram_lc_test.bash` - path-to-baseline, path-to-best-serialized-model, path-to-test-set, path-to-test-labels

- Eval Models - 
	``` (lang_idf_env)$ bash ngram_lc_test.bash save_best/baseline_classifier.pickle save_best/best_language_classifier.pickle merged_sets/test_set.conllu merged_sets/test_labels.txt``` 