from prompt_search import prompt_search_experiment

if __name__ == '__main__':
    #== Set Exp arguments (beforehand) ============================================================#
    datasets = ['snli', 'mnli']

    templates = [
        'is the second text an entailment of the first text? <t1> <t2>',
        'does the second text directly follow from the first text? <t1> <t2>',
        'are the texts related? <t1> <t2>' ,
        'are the texts consistent? <t1> <t2>',
        'does text 1 imply text 2? <t1> <t2>',
        'can text 2 be logically derived from text 1? <t1> <t2>',
        'does the hypothesis logically follow the premise? <t1> <t2>'
    ]

    label_word_sets = [
        ['yes', 'correct', 'yeah', 'follows'],
        ['maybe', 'unclear', 'perhaps', 'neutral'],
        ['no', 'incorrect', 'negative', 'contradiction']
    ]

    #== Run main experiment =======================================================================#
    prompt_search_experiment(
        datasets=datasets, 
        templates=templates, 
        label_word_sets=label_word_sets, 
    )
