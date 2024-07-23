import pandas as pd
data = pd.read_json('cladder.json')
data = data.dropna().reset_index(drop=True)

data = data.assign(step0 = lambda x: (x['reasoning']))
df = data['step0'].apply(lambda x: (x.get("step0")))
data['step0'] = df

data = data.assign(step1 = lambda x: (x['reasoning']))
df = data['step1'].apply(lambda x: (x.get("step1")))
data['step1'] = df

#step2) query type
data = data.assign(step2 = lambda x: (x['meta']))
df = data['step2'].apply(lambda x: (x.get("query_type")))
data['step2'] = df

#step3) formalize query
data = data.assign(step3 = lambda x: (x['reasoning']))
df = data['step3'].apply(lambda x: (x.get("step2")))
data['step3'] = df

#step4) extract all available data
data = data.assign(step4 = lambda x: (x['reasoning']))
df = data['step4'].apply(lambda x: (x.get("step4")))
data['step4'] = df

#step5) deduce estimand
data = data.assign(step5 = lambda x: (x['reasoning']))
df = data['step5'].apply(lambda x: (x.get("step3")))
data['step5'] = df

#step6)Insert the relevant data in Step 4 into the estimand, perform basic arithmetic calculations
data = data.assign(step6 = lambda x: (x['reasoning']))
df = data['step6'].apply(lambda x: (x.get("step5")))
data['step6'] = df

#end) derive the final answer
data = data.assign(end = lambda x: (x['reasoning']))
df = data['end'].apply(lambda x: (x.get("end")))
data['end'] = df

#########################################################################################################

#create a list of real variable names and their respective representation e.g. [['X','wife'],['Y','husband']]
def list_representation(x):
    x = x.get('step0')
    x = x[4:len(x)-1] #erase "Let " and the last "."
    x = x.split('; ')

    aux = []
    for i in range(len(x)):
        aux.append(x[i].split(" = "))
    return aux

#receives a list from list_representation and returns the prompt with it
def prompt_repres(x):
    x = x.get('step0')
    x = x[4:len(x)-1] #erase "Let " and the last "."
    x = x.split('; ')

    list_rep = []
    for i in range(len(x)):
        list_rep.append(x[i].split(" = "))
    
    prompt = ""
    for i in range(len(list_rep)):
        prompt += "Use "+list_rep[i][0]+" to represent "+list_rep[i][1]+". "
    return prompt

def create_string(str_list):
    string = ""
    for i in range(str_list):
        string += str_list[i]
    return string

#exchange query_type abbreviation to the whole word
def expand(abb):
    if abb == 'nde':
        return 'natural direct effect'
    elif abb == 'ate':
        return 'average treatment effect'
    elif abb == 'marginal':
        return 'marginal probability'
    elif abb == 'nie':
        return 'natural indirect effect'
    elif abb == 'det-counterfactual':
        return 'normal counterfactual question'
    elif abb == 'ett':
        return 'average treatment effect on treated'
    elif abb == 'correlation':
        return 'conditional probability'
    elif abb == 'exp_away':
        return 'explaining away effect'
    elif abb == 'collider_bias':
        return 'collider bias'
    elif abb == 'backadj':
        return 'backdoor adjustment set'
    return 0
    

data['step2'] = data.apply(lambda x: expand(x['step2']), axis=1)

###################################################################################################################3


#create columnn with complete prompt to be given to the LLM
paraphrases = [
            "Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships. ",
            "Think of a self-contained, hypothetical setting with just the specified conditions, and devoid of any unknown factors or causal connections. ",
            "Consider a self-contained, hypothetical world with solely the mentioned conditions, and is free from any hidden factors or cause-and-effect relationships. ",
            "Imagine a self-contained, hypothetical setting with merely the stated conditions, and absent any unmentioned factors or causative links. ",
            "Think of a self-contained, hypothetical world with only the given conditions, and is void of any unknown factors or causative relationships. ",
        ]

prompt_end0 = " Guidance: Address the question by following the steps below:\nStep 1) Extract the causal graph: Identify the causal graph that\
depicts the relationships in the scenario. "
prompt_end1 = "The diagram should simply consist of edges denoted in 'var1 -> var2' format, separated by commas. Step 2) Determine the query \
type: Identify the type of query implied by the main question. Choices include 'marginal probability', 'conditional probability', 'explaining\
away effect', 'backdoor adjustment set', 'average treatment effect', 'collider bias', 'normal counterfactual question', 'average treatment\
effect on treated', 'natural direct effect' or 'natural indirect effect'. Your answer should only be a term from the list above, enclosed \
in quotation marks. Step 3) Extract all the available data. Your answer should contain nothing but marginal probabilities and conditional \
probabilities in the form 'P(...)=...' or 'P(...|...)=...', each probability being separated by a semicolon. Stick to the previously \
mentioned denotations for the variables. Based on all the reasoning above, output <YES> or <NO> to answer the following question: "

'''"The diagram should simply consist of edges denoted in 'var1 -> var2' format, separated by commas. Step 2) Determine \
the query type: Identify the type of query implied by the main question. Choices include 'marginal probability', 'conditional probability', \
'explaining away effect', 'backdoor adjustment set', 'average treatment effect', 'collider bias', 'normal counterfactual question', \
'average treatment effect on treated', 'natural direct effect' or 'natural indirect effect'. Your answer should only be a term from the \
list above, enclosed in quotation marks. Step 3) Formalize the query: Translate the query into its formal mathematical expression based on \
its type, utilizing the 'do(·)' notation or counterfactual notations as needed. Step 4) Extract all the available data. Your answer should \
contain nothing but marginal probabilities and conditional probabilities in the form 'P(...)=...' or 'P(...|...)=...', each probability\
being separated by a semicolon. Stick to the previously mentioned denotations for the variables. Step 5) Given all the information above, \
deduce the estimand using skills such as do-calculus, counterfactual prediction, and the basics of probabilities. Answer step by step. \
Step 6) Insert the relevant data in Step 4 into the estimand, perform basic arithmetic calculations, and derive the final answer. There \
is an identifiable answer. Answer step by step. \nBased on all the reasoning above, output one word to answer the initial question."'''


data['prompt'] = data.apply(lambda x: (paraphrases[0] + 'Question: ' + x['given_info']+prompt_end0+prompt_repres(x['reasoning'])+prompt_end1+x['question']),axis=1)

####################################################################################################################################################
# Using open() function
file_path = "/media/data/bazaluk/cladder_octopusv4.csv"
	
print(f"File '{file_path}' created successfully.\n")

#generate answer for all the test data

query_type = ['marginal probability', 'conditional probability', 'explaining away effect', 'backdoor adjustment set', 
              'average treatment effect', 'collider bias', 'normal counterfactual question', 'average treatment effect on treated', 
              'natural direct effect', 'natural indirect effect']

data['predicted graph'] = None
data['predicted query type'] = None
data['predicted final answer'] = None
data['predicted text'] = None

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
torch.random.manual_seed(314159)

model = AutoModelForCausalLM.from_pretrained(
    "NexaAIDev/Octopus-v4", 
    device_map="cuda:0", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True 
)
tokenizer = AutoTokenizer.from_pretrained("NexaAIDev/Octopus-v4")

for k in range(len(data)):
#for k in range(2):
    question = data['prompt'].iloc[k]

    inputs = f"<|system|>You are a router. Below is the query from the users, please call the correct function and generate the parameters to call the function.<|end|><|user|>{question}<|end|><|assistant|>"

#print('\n============= Below is the response ==============\n')

# You should consider to use early stopping with <nexa_end> token to accelerate
    input_ids = tokenizer(inputs, return_tensors="pt")['input_ids'].to(model.device)

    generated_token_ids = []
#start = time.time()

# set a large enough number here to avoid insufficient length
    for i in range(500):
        next_token = model(input_ids).logits[:, -1].argmax(-1)
        generated_token_ids.append(next_token.item())
    
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=-1)

    # 32041 is the token id of <nexa_end>
        if next_token.item() == 32041:
            break

    text = tokenizer.decode(generated_token_ids)
    data.at[k, 'predicted text'] = text

    #check query_type
    for j in range(len(query_type)):
        if query_type[j] in text:
            data.at[k, 'predicted query type'] = query_type[j]

    #check graph
    import re

    if re.search('Step 1', text):
        step1_end = re.search('Step 1', text).end()
        if re.search('Step 2', text):
            step2_start = re.search('Step 2', text).start()
            data.at[k, 'predicted graph'] = text[step1_end:step2_start]
        else:
            data.at[k, 'predicted graph'] = None
    
    else:
        data.at[k, 'predicted graph'] = None
    
    #check final answer
    yes = len(re.findall('<yes>', text.lower()))
    no = len(re.findall('<no>', text.lower()))
    if yes >= no:
        data.at[k, 'predicted final answer'] = 'yes'
    else:
        data.at[k, 'predicted final answer'] = 'no'

    
    print(k)

data.to_csv(file_path, header=True, mode='a')
print("Finished.")
#print(generated_token_ids)
#end = time.time()
#print(f'Elapsed time: {end - start:.2f}s')
#data

































