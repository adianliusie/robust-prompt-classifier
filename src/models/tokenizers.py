from transformers import BertTokenizerFast, RobertaTokenizerFast, DebertaTokenizerFast, LlamaTokenizer, AutoTokenizer

def load_tokenizer(system:str)->AutoTokenizer:
    """ downloads and returns the relevant pretrained tokenizer from huggingface """
    if system   == 'bert-base'     : tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif system == 'bert-rand'     : tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif system == 'bert-large'    : tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")
    elif system == 'roberta-base'  : tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif system == 'roberta-large' : tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
    elif system == 'debert-base'   : tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
    elif system == 'deberta-large' : tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-large")
    elif system == 'deberta-xl'    : tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-xlarge")
    
    elif system == 'llama-7b'      : tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    elif system == 'alpaca-7b'     : tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
    elif system == 'opt-iml-1.3b'  : tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-1.3b")

    elif system == 't5-small'      : tokenizer = AutoTokenizer.from_pretrained("t5-small", return_dict=True)
    elif system == 't5-base'       : tokenizer = AutoTokenizer.from_pretrained("t5-base", return_dict=True)
    elif system == 't5-large'      : tokenizer = AutoTokenizer.from_pretrained("t5-large", return_dict=True)
    elif system == 't5-3b'         : tokenizer = AutoTokenizer.from_pretrained("t5-3b", return_dict=True) 
    elif system == 't5-11b'        : tokenizer = AutoTokenizer.from_pretrained("t5-11b", return_dict=True) 
    elif system == 'flan-t5-small' : tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", return_dict=True)
    elif system == 'flan-t5-base'  : tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", return_dict=True)
    elif system == 'flan-t5-large' : tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", return_dict=True)
    elif system == 'flan-t5-3b'    : tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl", return_dict=True) 
    elif system == 'flan-t5-11b'   : tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl", return_dict=True) 
    else: raise ValueError(f"invalid transfomer system provided: {system}")
    return tokenizer
       
