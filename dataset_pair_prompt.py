import pandas as pd
data = pd.read_json('cladder.json')
data = data.dropna().reset_index(drop=True)

data = data.assign(step0 = lambda x: (x['reasoning']))
df = data['step0'].apply(lambda x: (x.get("step0")))
data['step0'] = df

data = data.assign(step1 = lambda x: (x['reasoning']))
df = data['step1'].apply(lambda x: (x.get("step1")))
data['step1'] = df

#create a list of real variable names and their respective representation e.g. [['X','wife'],['Y','husband']]
def list_representation(x):
    x = x.get('step0')
    x = x[4:len(x)-1] #erase "Let " and the last "."
    x = x.split('; ')

    aux = []
    for i in range(len(x)):
        aux.append(x[i].split(" = "))
    return aux

def pair_questions(variables):
    variables = [i[0] for i in variables]
    pairs = []

    for i in range(len(variables)):
        for j in range(len(variables)):
            if i != j:
                pairs.append([variables[i],variables[j]])
    return pairs

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
        prompt += " Use "+list_rep[i][0]+" to represent "+list_rep[i][1]+"."
    return prompt

def prompt_questions(x):
    pairs = list_representation(x)
    pairs = pair_questions(pairs)
    prompt = ''
    for i in range(len(pairs)):
        p = ' ' + str(i+1) + ') Does '+pairs[i][0]+' cause '+pairs[i][1]+'?'
        prompt += p
    
    return prompt

pd.set_option('display.max_colwidth', None)
pair_suffix = ' Based on your answers, identify the causal graph that depicts the relationships in the scenario. The diagram should simply consist of edges denoted in "var1 -> var2" format, separated by commas.'
p_suffix = ' Identify the causal graph that depicts the relationships in the scenario. The diagram should simply consist of edges denoted in "var1 -> var2" format, separated by commas.'

data['prompt_pairs'] = data.apply(lambda x: (x['given_info']+prompt_repres(x['reasoning'])+prompt_questions(x['reasoning'])+pair_suffix),axis=1)
data['prompt'] =  data.apply(lambda x: (x['given_info']+prompt_repres(x['reasoning'])+p_suffix),axis=1)
data['prefix'] = data.apply(lambda x: (p_suffix+prompt_repres(x['reasoning'])),axis=1)

data = data[['prompt','prompt_pairs','prefix','step1']]
data = data[:5259].sample(frac=1).reset_index(drop=True)
data.to_csv('/media/data/bazaluk/data_pairs_gen.csv')