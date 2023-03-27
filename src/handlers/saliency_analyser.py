import torch
import copy

from captum.attr import Saliency, LayerIntegratedGradients
from captum.attr import visualization as viz
from types import MethodType

from ..evaluater import Evaluator
from ..models.mlm_prompting import TransformerModel

class InterpretableLoader(Evaluator):
    def __init__(self, exp_path:str):
        self.exp_path = exp_path
        self.dir = DirHelper.load_dir(exp_path)
        super().set_up_helpers()

    def saliency(self, data_name:str=None, mode:str='test', idx:int=0, ex=None, visible=True):
        #load example to run interpretability on
        assert (ex is None) or (data_name is None)
        if data_name:
            ex = self.get_ex(data_name=data_name, mode=mode, idx=idx) 

        #need to overwrite model, such that model  outputs logits
        self.model.__class__ = TransformerModelSaliency

        #get relevant input info    
        pred_scores = self.model(input_ids=ex.input_ids)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(ex.input_ids[0].tolist())

        print('ok up to here')
        saliency = Saliency(self.model)
        print('this is also okay...')

        attributions = saliency.attribute(ex.input_ids, target=1, abs=False)
        print('damn why do I never read this?')

        if visible:
            vis = viz.VisualizationDataRecord(
                    attributions,
                    torch.softmax(pred_scores, dim=0)[1],
                    torch.argmax(pred_scores),
                    str(ex.labels[0].item()),
                    torch.argmax(pred_scores),
                    attributions.sum(),       
                    tokens,
                    1)
            viz.visualize_text([vis])

        #rreturn model back to original class
        self.model.__class__ = TransformerModel

        return tokens, attributions 

    def integrad(self, data_name:str=None, mode:str='test', idx:int=0, ex=None, visible=True):
        #load example to run interpretability on
        assert (ex is None) or (data_name is None)
        if data_name:
            ex = self.get_ex(data_name=data_name, mode=mode, idx=idx) 
            
        #create baseline tensors
        baseline_ids = self.create_baseline(ex)

        #get relevant input info
        pred_scores = self.model(input_ids=ex.input_ids).logits[0]
        tokens = self.tokenizer.convert_ids_to_tokens(ex.input_ids[0].tolist())
        
        #set up the ingtegrated gradients
        lig = LayerIntegratedGradients(self.forward_interpret, self.model.transformer.embeddings.word_embeddings)

        attributions, delta = lig.attribute(inputs=ex.input_ids,
                                            baselines=baseline_ids,
                                            return_convergence_delta=True, 
                                            internal_batch_size=8)
        
        attributions = attributions.sum(dim=-1).squeeze(0)
        #attributions = attributions / torch.norm(attributions)

        #visualise the saliency attribution
        if visible:
            vis = viz.VisualizationDataRecord(
                    attributions,
                    torch.softmax(pred_scores, dim=0)[1],
                    torch.argmax(pred_scores),
                    str(ex.labels[0].item()),
                    torch.argmax(pred_scores),
                    attributions.sum(),       
                    tokens,
                    delta)
            viz.visualize_text([vis])

        return tokens, attributions             
        
    def forward_interpret(self, input_ids:torch.LongTensor):
        output = self.model(input_ids=input_ids)
        #y = output.logits.max(1).values
        y = output.logits[:,1]
        return y
    
    def create_baseline(self, ex):
        assert ex.input_ids.size(0) == 1
        pad_idx  = self.tokenizer.pad_token_id 
        format_tokens = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
        baseline_ids = [(tok if tok in format_tokens else pad_idx) for tok in ex.input_ids[0]]
        baseline_ids = torch.LongTensor([baseline_ids]).to(ex.input_ids.device)
        return baseline_ids
    
    def get_ex(self, data_name:str, mode:str, idx:int):
        data_set = self.data_loader.prep_split(data_name=data_name, mode=mode)
        data_set = list(self.batcher(data_set, bsz=1))
        ex = data_set[idx]
        #return Katherine
        return ex

    @property
    def tokenizer(self): 
        return self.data_loader.tokenizer

class TransformerModelSaliency(TransformerModel):
    def forward(self, *args, **kwargs):
        trans_output = self.transformer(*args, **kwargs)
        H = trans_output.last_hidden_state  #[bsz, L, 768] 
        h = H[:, 0]                         #[bsz, 768] 
        logits = self.output_head(h)        #[bsz, C] 
        return logits