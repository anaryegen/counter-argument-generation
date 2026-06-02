import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from conversation import get_conv_template
from tqdm import tqdm

#To use Llama 2 tokenizer, call huggingface-cli to login to account using token from https://huggingface.co/settings/tokens to get authentication to gated repo
tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-7b-v2.0")
model = AutoModelForCausalLM.from_pretrained("prometheus-eval/prometheus-7b-v2.0", device_map="auto")

# def run_eval(claim: str, counter_claim: str):

f = '../generated_output/cmdr_counter_argument.jsonl'
task_name = f.split('/')[-1].split('.')[0]
output_df = pd.DataFrame(columns=['argument', 'counter-argument',task_name + '_opposition_score', task_name + '_opposition_exp', 
            task_name + '_relatedness_score', task_name + '_relatedness_exp', task_name + '_specificity_score', 
            task_name + '_specificity_exp', task_name + '_factuality_score', task_name + '_factuality_exp', task_name + '_fluency_score',
            task_name + '_fluency_exp', task_name + '_persuasive_score', task_name + '_persuasive_exp'])
    
aspect_list = ["Opposition", "Relatedness", "Specificity", "Factuality", "Fluency", "Persuasive"]
opp_score, opp_exp, rel_score, rel_exp, spe_score, spe_exp, fac_score, fac_exp, flu_score, flu_exp, per_score, per_exp = [], [], [], [], [], [], [], [], [], [], [], []
            

prompts = pd.read_csv("prompts.csv")

claims = [eval(l)['argument'] for l in open(f).readlines()]
counter_claims = [eval(l)['model'] for l in open(f).readlines()]
assert len(claims) == len(counter_claims), f'The length of claim and counter-claim are different. Claim: {len(claims)} and Counter-claim: {len(counter_claims)}'


# JSON file containing the score rubrics and reference answers for each CN evaluation aspect
# prompt_text = pd.read_json("/gaueko0/users/ayeginbergenov001/rag-arg/evaluation/Prometheus/cn_prometheus.json")

# Task description and feedback portions of the Prometheus evaluation prompt
task_description = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 3, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 3. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 3)\"
4. Please do not generate any other opening, closing, and explanations.

"""
feedback = """\n\n###Feedback: """

for i in tqdm(range(len(claims))):
    # Fills template for instruction and response in Prometheus evaluation prompt for each pair
    instruction = """###The instruction to evaluate:\nGenerate a counter argument response to the following claim example.\nClaim: """ + claims[i]

    response = """\n\n###Response to evaluate:\nCounter Argument: """ + counter_claims[i]

    for aspect in aspect_list:
        # Fills template for reference answer and score rubric in Prometheus evaluation prompt for each evaluation aspect
        # reference_answer = """\n\n###Reference Answer (Score 3): """ + prompt_text["references"][aspect]

        # score_rubric = """\n\n###Score Rubrics: """ + prompt_text["rubrics"][aspect]
        score_rubric = """\n\n###Score Rubrics: """ + prompts[aspect][0]

        # input_text = task_description + instruction + response + reference_answer + score_rubric + feedback
        input_text = task_description + instruction + response + score_rubric + feedback

        # Defines use of llama-2 conv template for Prometheus instructions
        conv = get_conv_template("mistral")

        # Sets system message and input prompt used for Prometheus evaluation within the conv template
        conv.set_system_message("You are a fair evaluator language model.")
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda") 

        # attention_mask = input_ids["attention_mask"]

        outputs = model.generate(input_ids, do_sample=True, temperature=0.1, top_p=0.9, max_new_tokens=236, repetition_penalty=1.03, pad_token_id=tokenizer.eos_token_id)

        # Pulls score and explanation from conv template 
        decoded = tokenizer.decode(outputs[0])
        x = decoded.split('[/INST]')
        y = x[1].split('[RESULT] ')
        explanation = y[0]
        if len(y) <= 1:
            score = "N/A"
        else:
            z = y[1].split('</s>')
            score = z[0]

        if aspect == "Opposition":
            opp_score.append(score)
            opp_exp.append(explanation)
        elif aspect == "Relatedness":
            rel_score.append(score)
            rel_exp.append(explanation)
        elif aspect == "Specificity":
            spe_score.append(score)
            spe_exp.append(explanation)
        elif aspect == "Factuality":
            fac_score.append(score)
            fac_exp.append(explanation)
        elif aspect == "Fluency":
            flu_score.append(score)
            flu_exp.append(explanation)
        elif aspect == "Persuasive":
            per_score.append(score)
            per_exp.append(explanation)

# Updating output dataframe with lists containing each generated evaluation score and explanation
output_df['argument'] = claims
output_df['counter-argument'] = counter_claims

output_df[task_name + '_opposition_score'] = opp_score
output_df[task_name + '_opposition_exp'] = opp_exp
    
output_df[task_name + '_relatedness_score'] = rel_score
output_df[task_name + '_relatedness_exp'] = rel_exp
        
output_df[task_name + '_specificity_score'] = spe_score
output_df[task_name + '_specificity_exp'] = spe_exp

output_df[task_name + '_factuality_score'] = fac_score
output_df[task_name + '_factuality_exp'] = fac_exp

output_df[task_name + '_fluency_score'] = flu_score
output_df[task_name + '_fluency_exp'] = flu_exp

output_df[task_name + '_persuasive_score'] = per_score
output_df[task_name + '_persuasive_exp'] = per_exp

output_df.to_csv(task_name + "_scores_prometheus2.csv", index=False)
output_df = output_df[pd.to_numeric(output_df[task_name + '_opposition_score'], errors='coerce').notnull()]
output_df = output_df[pd.to_numeric(output_df[task_name + '_factuality_score'], errors='coerce').notnull()]
output_df = output_df[pd.to_numeric(output_df[task_name + '_relatedness_score'], errors='coerce').notnull()]
output_df = output_df[pd.to_numeric(output_df[task_name + '_specificity_score'], errors='coerce').notnull()]
output_df = output_df[pd.to_numeric(output_df[task_name + '_persuasive_score'], errors='coerce').notnull()]

print(f"Evaluated file: {f}")
print('All scores: ')
print("Opposition: ", output_df[task_name + '_opposition_score'].tolist())
print("Factuality: ", output_df[task_name + '_factuality_score'].tolist())
print("Relatedness: ", output_df[task_name + '_relatedness_score'].tolist())
print("Specificity: ", output_df[task_name + '_specificity_score'].tolist())
print("Persuasive: ", output_df[task_name + '_persuasive_score'].tolist())

print("Avg. Opposition score: ", output_df[task_name + '_opposition_score'].apply(lambda x: int(x)).mean())
print("Avg. Factuality score: ", output_df[task_name + '_factuality_score'].apply(lambda x: int(x)).mean())
print("Avg. Relatedness score: ", output_df[task_name + '_relatedness_score'].apply(lambda x: int(x)).mean())
print("Avg. Specificity score: ", output_df[task_name + '_specificity_score'].apply(lambda x: int(x)).mean())
print("Avg. Persuasive score: ", output_df[task_name + '_persuasive_score'].apply(lambda x: int(x)).mean())

sum_opp = output_df[task_name + '_opposition_score'].apply(lambda x: int(x)).sum()
sum_fact =  output_df[task_name + '_factuality_score'].apply(lambda x: int(x)).sum()
sum_rel = output_df[task_name + '_relatedness_score'].apply(lambda x: int(x)).sum()
sum_spec = output_df[task_name + '_specificity_score'].apply(lambda x: int(x)).sum()
sum_pers = output_df[task_name + '_persuasive_score'].apply(lambda x: int(x)).sum()

print('Sum Opposition: ', sum_opp)
print('Sum Factuality: ', sum_fact)
print('Sum Relatedness: ', sum_rel)
print('Sum Specificty: ', sum_spec)
print('Sum Persuasiveness: ', sum_pers)



