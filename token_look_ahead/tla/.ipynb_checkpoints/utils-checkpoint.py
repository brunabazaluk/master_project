import torch
import numpy as np
from transformers import LogitsProcessor

torch.set_float32_matmul_precision('high')

class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self, dfa):

        self.dfa = dfa
        self.dfa['current_state'] = 0
    
    def __call__(self, input_ids, scores):

        #put to -inf the logits that are not accepted by the DFA
        edges_list = self.dfa.get('edges')
        current_state = self.dfa.get('current_state')
    
        possible_edges = [e for e in edges_list if e[0] == current_state]
        
        possible_edges_ = np.zeros(len(possible_edges[0][2]), dtype=bool)
        
        for i in range(len(possible_edges)):
            possible_edges_ = possible_edges_ + possible_edges[i][2]


        #eos is decoded as 2
        if current_state in self.dfa.get('accept_states'):
            possible_edges_[2] = True
            #print([i for i, x in enumerate(possible_edges_) if x])
        
        scores[0] = torch.where(torch.tensor(~possible_edges_, device=torch.device('cuda')), torch.tensor(-1e30, device=torch.device('cuda')),scores[0])

        #set new current state
        next_token_ = torch.argmax(scores[0])
        #print(next_token_)
        for e in possible_edges:
            if e[2][next_token_]: #this is the chosen edge
                #set new current state to dfa
                self.dfa['current_state'] = e[1]
                #print(e[1])
                break
        
        return scores


def extract_generated_ids(outputs, prompt_ids, suffix_ids, eos_token_id):
    processed_outputs = []

    suffix_ids = tuple(suffix_ids)
    while len(suffix_ids) > 0 and suffix_ids[-1] == eos_token_id:
        suffix_ids = suffix_ids[:-1]
    prompt_ids = tuple(prompt_ids)

    for output_ids in outputs:
        output_ids = tuple(output_ids)
        while output_ids[-1] == eos_token_id:
            output_ids = output_ids[:-1]
        output_ids = output_ids[len(prompt_ids):]

        l = 0
        for k in range(1, min(len(output_ids), len(suffix_ids))+1):
            if output_ids[-k:] == suffix_ids[:k]:
                l = k
        end = None if l == 0 else -l

        output_ids = output_ids[:end]

        processed_outputs.append(output_ids)

    return processed_outputs


def populate_edge(edge_vocab=[], vocab_size=0, tokenizer=None, ALL=False): #edge_vocab: list of all accepted words for that edge
    if (ALL):
        return np.ones((vocab_size,), dtype=bool)
    else:
        edge_set = np.zeros((vocab_size,), dtype=bool)
        edge_tokens = []

        for i in range(len(edge_vocab)):
            edge_set[tokenizer.encode(edge_vocab[i], add_special_tokens=False)] = 1

    return edge_set