import pandas as pd
from utils import *


dataset = 'cweb'
method = 'pipeline'
kg = 'yago'
model_name = 'gpt35'
candidateSet = 'chatel'

df = pd.read_csv(r'./results_'+kg+'/'+dataset+'_'+method+'_'+model_name+'_'+kg+'_'+candidateSet)
df.columns = ['id','candidate_length','in_candidates','response', 'answer']
total_instances = len(df)
zero_candidates = len(df[df['candidate_length']==0])
not_in_candidates = len(df[df['in_candidates']==False])
not_in_candidates_no_zero = len(df[(df['in_candidates']==False)&(df['candidate_length']!=0)])


# Remove notInKB
df = df[df['candidate_length']!=0]
df_filtered = df[df['response']!='notInKB'] 


len_filtered = len(df_filtered)
not_in_c = len(df_filtered[df_filtered['in_candidates']==False])
TP = sum(df_filtered['answer'] == clean(df_filtered['response']))
FP = len_filtered - TP
FN = len_filtered - TP

accuracy = TP/len_filtered
micro_f1 = TP/(TP+0.5*(FP+FN))
gold = (len_filtered-not_in_c)/len_filtered

print('inKB micro F1:', micro_f1)
print('Gold-F1:',gold)
print('Gold Percentage:', micro_f1/gold)











