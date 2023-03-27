from prompt_search import prompt_search_experiment

if __name__ == '__main__':
    #== Set Exp arguments (beforehand) ============================================================#
    datasets = ['sst', 'rt', 'imdb', 'amazon-s', 'yelp-s']

    templates = [
        'classify the following review: <t>',
        'how was the movie? <t>',
        'which word best describes the text? <t>',
        'what is the sentiment? <t>',
        "what is the reviewer's verdict? <t>",
        'is the following movie good or bad? <t>'
    ]

    label_word_sets = [
        ['bad', 'terrible', 'poor', 'horrible', 'negative'],
        ['good', 'great', 'amazing', 'fantastic', 'positive'],
    ]

    #== Run main experiment =======================================================================#
    prompt_search_experiment(
        datasets=datasets, 
        templates=templates, 
        label_word_sets=label_word_sets, 
        save_probs=False
    )
