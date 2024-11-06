import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import pandas as pd
from conversation import get_conv_template
from tqdm import tqdm

#To use Llama 2 tokenizer, call huggingface-cli to login to account using token from https://huggingface.co/settings/tokens to get authentication to gated repo
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("kaist-ai/Prometheus-13b-v1.0", device_map="auto")

# def run_eval(claim: str, counter_claim: str):

task_name = "cmv_rag_qa"
output_df = pd.DataFrame(columns=['argument', 'counter-argument',task_name + '_opposition_score', task_name + '_opposition_exp', 
            task_name + '_relatedness_score', task_name + '_relatedness_exp', task_name + '_specificity_score', 
            task_name + '_specificity_exp', task_name + '_factuality_score', task_name + '_factuality_exp', task_name + '_fluency_score',
            task_name + '_fluency_exp', task_name + '_persuasive_score', task_name + '_persuasive_exp'])
aspect_list = ["Opposition", "Relatedness", "Specificity", "Factuality", "Fluency", "Persuasive"]
opp_score, opp_exp, rel_score, rel_exp, spe_score, spe_exp, fac_score, fac_exp, flu_score, flu_exp, per_score, per_exp = [], [], [], [], [], [], [], [], [], [], [], []
            

f = open('../../generated_output/cmv_rag_qa.txt').readlines()
# df = pd.read_csv('/gaueko0/users/ayeginbergenov001/rag-arg/change_my_view_cohere_distilled.csv')
prompts = pd.read_csv("../prompts.csv")
claims = []
counter_claims = []
# df = df.drop_duplicates(['model'])
# print('SHAPE: ', df.shape)
# hs = df['argument'].to_list()
# cn = df['counter-argument'].to_list()

for line in f:
    line = eval(line)
    claims.append(line['argument'])
    #counter_claims.append(line['model'])
    counter_claims.append(line['cmdr_websearch'])


# JSON file containing the score rubrics and reference answers for each CN evaluation aspect
prompt_text = pd.read_json("cn_prometheus.json")

# Task description and feedback portions of the Prometheus evaluation prompt
task_description = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

"""
feedback = """\n\n###Feedback: """

for i in tqdm(range(len(claims))):
    # Fills template for instruction and response in Prometheus evaluation prompt for each HS/CN pair
    instruction = """###The instruction to evaluate:\nGenerate a counter argument response to the following claim example.\nClaim: """ + claims[i]

    response = """\n\n###Response to evaluate:\nCounter Argument: """ + counter_claims[i]

    for aspect in aspect_list:
        # Fills template for reference answer and score rubric in Prometheus evaluation prompt for each evaluation aspect
        reference_answer = """\n\n###Reference Answer (Score 5): """ + prompt_text["references"][aspect]

        score_rubric = """\n\n###Score Rubrics: """ + prompt_text["rubrics"][aspect]

        input_text = task_description + instruction + response + reference_answer + score_rubric + feedback

        # Defines use of llama-2 conv template for Prometheus instructions
        conv = get_conv_template("llama-2")

        # Sets system message and input prompt used for Prometheus evaluation within the llama-2 conv template
        conv.set_system_message("You are a fair evaluator language model.")
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda") 

        # attention_mask = input_ids["attention_mask"]

        outputs = model.generate(input_ids, do_sample=True, temperature=0.1, top_p=0.9, max_new_tokens=256, repetition_penalty=1.03, pad_token_id=tokenizer.eos_token_id)

        # Pulls score and explanation from llama-2 conv template 
        decoded = tokenizer.decode(outputs[0])
        x = decoded.split('[/INST]  ')
        y = x[1].split(' [RESULT] ')
        explanation = y[0]
        if len(y) <= 1:
            score = 0
        else:
            z = y[1].split('</s>')
            score = z[0]
        # print('Feedback: ' + explanation)
        # print('Score: ' + score)
        
    #        if not isinstance(score, int):
    #            score = 0
    #        else: 
    #            score = float(score)
        # Adds each score and explanation to list for each aspect that will be used to update output dataframe 
        #if len(score) == 0:
        #    score = 0
        #else:
        #    score = score[0]
        #print(my_output + '\n')

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
# print(opp_score, rel_score, spe_score, fac_score, flu_score)
# return opp_score, opp_exp, rel_score, rel_exp, spe_score, spe_exp, fac_score, fac_exp, flu_score, flu_exp
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

output_df.to_csv(task_name + "_scores_prometheus.csv", index=False)
output_df = output_df[pd.to_numeric(output_df[task_name + '_opposition_score'], errors='coerce').notnull()]
output_df = output_df[pd.to_numeric(output_df[task_name + '_factuality_score'], errors='coerce').notnull()]
output_df = output_df[pd.to_numeric(output_df[task_name + '_relatedness_score'], errors='coerce').notnull()]
output_df = output_df[pd.to_numeric(output_df[task_name + '_specificity_score'], errors='coerce').notnull()]
output_df = output_df[pd.to_numeric(output_df[task_name + '_persuasive_score'], errors='coerce').notnull()]

print("Opposition score: ", output_df[task_name + '_opposition_score'].apply(lambda x: int(x)).mean())
print("Factuality score: ", output_df[task_name + '_factuality_score'].apply(lambda x: int(x)).mean())
print("Relatedness score: ", output_df[task_name + '_relatedness_score'].apply(lambda x: int(x)).mean())
print("Specificity score: ", output_df[task_name + '_specificity_score'].apply(lambda x: int(x)).mean())
print("Persuasive score: ", output_df[task_name + '_persuasive_score'].apply(lambda x: int(x)).mean())
# f = [eval(l) for l in open('../../mistral-counter-noqa-conan.jsonl').readlines()]  
# prometheus_eval_scores = {
#                     'opp_score' : [],
#                     'rel_score': [],
#                     'spe_score': [],
#                     'fac_score': [],
#                     'flu_score': [],
#                     'ovr_score': []
#                     }

# prometheus_eval_exps = {
#         'opp_exp': [],
#         'rel_exp': [],
#         'spe_exp': [],
#         'fac_exp': [],
#         'flu_exp': [],
#         'ovr_exp': []
#                 }
# for l in tqdm(f):
#     opp_score, opp_exp, rel_score, rel_exp, spe_score, spe_exp, fac_score, fac_exp, flu_score, flu_exp = run_eval(l['argument'], l['llama'])
#     prometheus_eval_scores['opp_score'].append(opp_score)
#     prometheus_eval_scores['rel_score'].append(rel_score)
#     prometheus_eval_scores['spe_score'].append(spe_score)
#     prometheus_eval_scores['fac_score'].append(fac_score)
#     prometheus_eval_scores['flu_score'].append(flu_score)

#     prometheus_eval_exps['rel_exp'].append(rel_exp)
#     prometheus_eval_exps['opp_exp'].append(opp_exp)
#     prometheus_eval_exps['spe_exp'].append(spe_exp)
#     prometheus_eval_exps['fac_exp'].append(fac_exp)
#     prometheus_eval_exps['flu_exp'].append(flu_exp)

# print('According to the Prometheus-as-a-Judge the scores are: ')
# print('Opposition score: ', sum(prometheus_eval_scores['opp_score']) / len(prometheus_eval_scores['opp_score']))
# print('Relatedness score: ', sum(prometheus_eval_scores['rel_score']) / len(prometheus_eval_scores['rel_score']))
# print('Specificity score: ', sum(prometheus_eval_scores['spe_score']) / len(prometheus_eval_scores['spe_score']))
# print('Factuality score: ', sum(prometheus_eval_scores['fac_score']) / len(prometheus_eval_scores['fac_score']))
# print('Fluency score: ', sum(prometheus_eval_scores['flu_score']) / len(prometheus_eval_scores['flu_score']))

# eval_results = open('eval_results.txt', 'w')
# eval_results.write("Prometheus scores: \n" + str(prometheus_eval_scores))
# # eval("The absence of free will can be logically inferred from various disciplines.", "The presence of free will is supported by our strong sense of freedom and the ability to make choices that are not determined by past events. The subjective experience of making choices and our sense of agency further support the idea of free will. Alternative explanations within these disciplines also challenge the notion of a lack of free will.")
