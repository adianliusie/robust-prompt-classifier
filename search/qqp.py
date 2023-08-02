from prompt_search import prompt_search_experiment

if __name__ == '__main__':
    #== Set Exp arguments (beforehand) ============================================================#
    datasets = ['qqp', 'mrpc']

    templates = [
        'is the second text a paraphrase of the first text? <t1> <t2>',
        'are the two texts semantically equivalent? <t1> <t2>',
        'are the texts paraphrases of each other? <t1> <t2>' ,
        'do the texts have the same meaning? <t1> <t2>',
        'is the meaning of text 1 the same as in text 2? <t1> <t2>',
        'would the two texts be classified as paraphrases? <t1> <t2>',
    ]

    label_word_sets = [
        ['no', 'incorrect', 'not', 'negative', 'false'],
        ['yes', 'correct', 'yeah', 'positive', 'true']
    ]

    #== Run main experiment =======================================================================#
    prompt_search_experiment(
        datasets=datasets, 
        templates=templates, 
        label_word_sets=label_word_sets, 
    )
