from transformers import LlamaForCausalLM, LlamaTokenizer
from fastchat.model.model_adapter import load_model
import pandas as pd
from peft import PeftConfig, PeftModel
from tqdm import tqdm

# Loads data of different data types to GPU 
def load2gpu(x, device):
    if x is None:
        return x
    if isinstance(x, dict):
        t2 = {}
        for key, val in x.items():
            t2[key] = val.to(device)
        return t2
    if isinstance(x, list):
        y = []
        for v in x:
            y.append(v.to(device))
        return y
    return x.to(device)

# Returns evaluation prompt based on given evaluation aspect
def prompt(aspect):
#    prompts = pd.read_csv("prompts.csv")
    if aspect == "Opposition":
        return prompts['Opposition'][0]
    elif aspect == "Relatedness":
        return prompts['Relatedness'][0]
    elif aspect == "Specificity":
        return prompts['Specificity'][0]
    elif aspect == "Factuality":
        return prompts["Factuality"][0]
    elif aspect == "Fluency":
        return prompts["Fluency"][0]
    elif aspect == "Persuasive":
        return prompts["Persuasive"][0]
    


generation_model = "cmv-rag-qa" # generation_model must be dialoGPT, chatGPT, or vicuna
output_df = pd.DataFrame(columns=['argument', 'counter-argument',generation_model + '_opposition_score', generation_model + '_opposition_exp', generation_model + '_relatedness_score', 
                    generation_model + '_relatedness_exp', generation_model + '_specificity_score', generation_model + '_specificity_exp', generation_model + '_factuality_score', 
                    generation_model + '_factuality_exp', generation_model + '_fluency_score', generation_model + '_fluency_exp', generation_model + '_persuasive_score', 
                    generation_model + '_persuasive_exp'])

# List used to trace through each evaluation aspect 
aspect_list = ["Opposition", "Relatedness", "Specificity", "Factuality", "Fluency", "Persuasive"]

# Lists used to store evaluation scores and explanations for each evaluation aspect 
opp_score, opp_exp, rel_score, rel_exp, spe_score, spe_exp, fac_score, fac_exp, flu_score, flu_exp, per_score, per_exp = [], [], [], [], [], [], [], [], [], [], [], []

#df.drop_duplicates(['argument'],inplace=True)
#f = open('/gaueko0/users/ayeginbergenov001/rag-arg/data/data.jsonl').readlines()
f = open("../../generated_output/cmv_rag_qa.txt").readlines()

# When the data is DataFrame
#hs = df['argument'].tolist()
#cn = df['counter-argument'].tolist()
claims, counter_claims = [], []
prompts = pd.read_csv('../prompts.csv')
# When the data is jsonl/txt
for line in f:
    line = eval(line)
    claims.append(line['argument'])
    counter_claims.append(line['cmdr_websearch'])



tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-33b-v1.3", legacy=False)
model, _ = load_model(model_path="lmsys/vicuna-33b-v1.3", device="cuda", num_gpus=2)

for i in tqdm(range(0, len(claims))):
    content = "Argument: " + claims[i] + "\nCounter-Argument: " + counter_claims[i] + "\nScore: "
    for aspect in aspect_list:
        p = prompt(aspect)
        instruction = p + "\n" + content

        inputs = tokenizer([instruction], return_tensors="pt", padding=True)
        input_ids = load2gpu(inputs['input_ids'], 'cuda')
        mask = load2gpu(inputs['attention_mask'], 'cuda')

        output = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=512, temperature=0)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response = decoded.replace(instruction, "")
        my_output = decoded.replace(p, "")
        score = [int(i) for i in response.split() if i.isdigit()]

        # If statement to check for instances where Vicuna doesn't generate a parsable evaluation score 
        if len(score) == 0:
            score = 0
        else:
            score = score[0]
        #print(my_output + '\n')
        
        if aspect == "Opposition":
            opp_score.append(score)
            opp_exp.append(response)
        elif aspect == "Relatedness":
            rel_score.append(score)
            rel_exp.append(response)
        elif aspect == "Specificity":
            spe_score.append(score)
            spe_exp.append(response)
        elif aspect == "Factuality":
            fac_score.append(score)
            fac_exp.append(response)
        elif aspect == "Fluency":
            flu_score.append(score)
            flu_exp.append(response)
        elif aspect == "Persuasive":
            per_score.append(score)
            per_exp.append(response)

# # Updating output dataframe with lists containing each generated evaluation score and explanation
output_df['argument'] = claims
output_df['counter-argument'] = counter_claims
output_df[generation_model + '_opposition_score'] = opp_score
output_df[generation_model + '_opposition_exp'] = opp_exp
    
output_df[generation_model + '_relatedness_score'] = rel_score
output_df[generation_model + '_relatedness_exp'] = rel_exp
        
output_df[generation_model + '_specificity_score'] = spe_score
output_df[generation_model + '_specificity_exp'] = spe_exp

output_df[generation_model + '_factuality_score'] = fac_score
output_df[generation_model + '_factuality_exp'] = fac_exp

output_df[generation_model + '_fluency_score'] = flu_score
output_df[generation_model + '_fluency_exp'] = flu_exp

output_df[generation_model + '_persuasive_score'] = per_score
output_df[generation_model + '_persuasive_exp'] = per_exp

output_df.to_csv(generation_model + "_scores_vicuna-v1.3.csv", index=False)

output_df = output_df[pd.to_numeric(output_df[generation_model + '_opposition_score'], errors='coerce').notnull()]
output_df = output_df[pd.to_numeric(output_df[generation_model + '_factuality_score'], errors='coerce').notnull()]
output_df = output_df[pd.to_numeric(output_df[generation_model + '_relatedness_score'], errors='coerce').notnull()]
output_df = output_df[pd.to_numeric(output_df[generation_model + '_specificity_score'], errors='coerce').notnull()]
output_df = output_df[pd.to_numeric(output_df[generation_model + '_persuasive_score'], errors='coerce').notnull()]

print("Opposition score: ", output_df[generation_model + '_opposition_score'].apply(lambda x: int(x)).mean())
print("Factuality score: ", output_df[generation_model + '_factuality_score'].apply(lambda x: int(x)).mean())
print("Relatedness score: ", output_df[generation_model + '_relatedness_score'].apply(lambda x: int(x)).mean())
print("Specificity score: ", output_df[generation_model + '_specificity_score'].apply(lambda x: int(x)).mean())
print("Persuasive score: ", output_df[generation_model + '_persuasive_score'].apply(lambda x: int(x)).mean())
