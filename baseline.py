import json
import csv
import os
from openai import OpenAI
from utils import *
from mistralai.client import MistralClient



dataset_list = ['wiki','aqu','ace2004','cweb','KORE50','msn','oke15','oke16','reu','RSS ']
os.makedirs('./results_baseline')

for dataset_name in dataset_list:

    # Experiment variables
    llm_provider = 'openai' #llmstudio, openai, mistral
    model_name = 'gpt35' #mistral-large, mistral-small, gpt35, mistral7B
    ontology = 'db' #yago
    candidateSet = 'chatel' # chatel
    dataset = dataset_name

    model = "gpt-3.5-turbo-1106"  #'mistral-large-latest' 


    if llm_provider == 'openai':
        client = OpenAI(api_key="yourOpenAIkey")
    elif llm_provider == 'llmstudio':
        client = OpenAI(base_url="yourLLMStudioHost", api_key="not-needed")
    elif llm_provider == 'mistral':
        client = MistralClient(api_key='yourMistralAPIkey')



    # Open mentions
    with open(r'./data/'+dataset+'_'+candidateSet+'.jsonl', 'r', encoding='utf-8') as file:

        with open(r'./results_baseline/'+dataset+'_baseline_'+model_name+'_'+ontology+'_'+candidateSet+'.csv', 'a', newline='', encoding='utf-8') as csvfile:

            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['id', 'candidate_length', 'in_candidates', 'response', 'answer'])

            for line in file: 
                data = json.loads(line) 
                id = data['id']
                candidates = data['candidates']
                text = data['input']
                answer = data['answer']
                len_candidates = len(candidates)
                answer_in_candidates = answer in candidates
                

                if not os.path.isfile(r'./subgraphs'+'_'+ontology+'/'+dataset+'_'+ontology+'_'+candidateSet+'/'+str(id)+'.pkl'):
                    csvwriter.writerow([id, len_candidates, answer_in_candidates, 'notInKB', 'notInKB'])
                    continue
                if len_candidates == 1:
                    csvwriter.writerow([id, len_candidates, answer_in_candidates, candidates[0], answer])
                elif not answer_in_candidates:
                    csvwriter.writerow([id, len_candidates, answer_in_candidates, 'nic', answer])
                else:
                    prompt = 'This is an Entity Disambiguation task. \nGiven the entity between [START_ENT] and [END_ENT]\n'+text
                    prompt = prompt +'\n\nTo which of these entities the mention ' + data['mention'] + ' refers to?\n'
                    for candidate in candidates:
                        prompt = prompt +candidate+'\n'
                    prompt += '\nProvide only the name of the entity exactly as provided in the list, in json format: {"entity": "entityName"}'
                    
                    
                    response = json.loads(get_response(prompt, client, provider= llm_provider, model = model))['entity']

                    csvwriter.writerow([id, len_candidates, answer_in_candidates, response, answer])
