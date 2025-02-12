# from datasets import load_dataset
import glob
import pandas as pd
import time
import json
import argparse 
import random

import httpx
from tqdm import tqdm
import cohere

from datasets import load_dataset


if __name__ == '__main__':
        
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--api_key",
            type=str,
            required=True,
            help="API key for Cohere",
        )
        parser.add_argument(
            "--data_name",
            type=str,
            default='candela',
            choices=['cmv', 'conan', 'persuasive', 'opinion', 'candela'],
            help="Path to the data with claims to counter",
            # nargs='+'
        )
        parser.add_argument(
            "--data_path",
            type=str,
            default='data/data.jsonl',
            help="Path to the data with claims to counter",
        )
        parser.add_argument(
            "--output_file",
            type=str,
            required=True,
            help="Path to save the generated data",
        )
        parser.add_argument(
            "--get_external_knowledge",
            type=str,
            default=False,
            help="Path to save the generated data",
        )
        args = parser.parse_args()

def rag(data_name: str):
    
    argument = []
    counter_argument = []
    if data_name == 'cmv':
        res_file = args.output_file
        data = [eval(l) for l in open(args.data_path).readlines()]
        for l in data:
            argument.append(l['argument'].strip())  
            counter_argument.append(l['counter-argument'].strip())
    
    if data_name == 'candela':
        res_file = args.output_file
        argument = open("candela_data/argument_150.txt").readlines()
        counter_argument = open("candela_data/counter_150.txt").readlines()

    if data_name =='conan':
        print('LOADING DATA: ')
        res_file = args.output_file
        data = load_dataset("HiTZ/CONAN-EUS", 'en')
        data = pd.DataFrame(data['test']).drop_duplicates(['HS'])
        argument = data['HS'].to_list()
        counter_argument = data['CN'].to_list()


    if data_name == 'persuasive':
        res_file = args.output_file
        data = load_dataset("Anthropic/persuasion")
        data = pd.DataFrame(data['train'])
        argument = data[data['rating_final'] == '1 - Strongly oppose'].drop_duplicates('claim')['claim'].to_list()
        argument += data[data['rating_final'] == '2 - Oppose'].drop_duplicates('claim')['claim'].to_list()
        counter_argument = data[data['rating_final'] == '1 - Strongly oppose'].drop_duplicates('claim')['argument'].to_list()
        counter_argument += data[data['rating_final'] == '2 - Oppose'].drop_duplicates('claim')['argument'].to_list()
        argument = argument
        counter_argument = counter_argument

    if data_name == 'opinion':
        res_file = args.output_file
        data = [eval(l) for l in open('mistral-opinions-norag-qa.jsonl').readlines()]
        argument = [] 
        for l in data:
            argument.append(l['argument'].strip())
        counter_argument = argument # temporal

    
    print("\tNumber of claims: " , len(argument))
    print("\tNumber of counter-claims: " , len(counter_argument))
    assert len(argument) == len(counter_argument), "The length of argument and counter-argument are different. Please check your data."




    # count = 0
    results = open(res_file , 'w')
    for claim, counter_claim in tqdm(zip(argument, counter_argument), total=len(argument)):
        # count += 1
        # if count % 10 == 0:
            # to comply with the 10 API calls/min restriction
            # time.sleep(60)
        questions = get_questions(claim)
        if args.get_external_knowledge:
            qacontext = get_qacontext(questions)
            counter, qacontext = get_counter_claim(claim, qacontext=qacontext)         
            output = {
                    'argument': claim, # input claim
                    'cmdr_websearch': counter, # generated counter-claim using external knowledge
                    'questions': questions, # generated questions from the input claim
                    'qa_context': qacontext, # answers to the generated questions using web-search 
                    'counter-argument': counter_claim # "gold standard" counter-claim
                }      
        else:
            counter, _ = get_counter_claim(claim, qacontext=None) 
            output = {
                    'argument': claim, # input claim
                    'cmdr': counter, # generated counter-claim 
                    'counter-argument': counter_claim # "gold" counter-claim
                }
        
        json.dump(output, results)
        results.write('\n')


def get_questions(claim: str):
    co = cohere.ClientV2(args.api_key)
    text = co.chat(
            model = 'command-r-plus',
            messages = [{
                "role": "user",
                "content": f"""Generate list of 5 queries for web-search that will help to find information to question the veracity of the given claim and persuade to take the opposing position: {claim}? 
                                Provide only questions and nothing else. 
                                Answer should be in JSON and stored under the question key."""
                }
            ],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                    "question": {"type": "array"},
                    },
                "required": ["question"],
        },
    },
        )
    

    
    questions = eval(text.message.content[0].text)
    return questions

def get_qacontext(questions: dict):
    # questions = eval(questions)
    context = []
    co = cohere.Client(args.api_key)
    if not questions:
        print("Questions were not provided, make sure that the logic of the function call is correct :)")
    else:
    
        if 'question' in questions:
            for question in questions['question']:
                response = co.chat(
                    model = 'command-r-plus',
                    message = f"""Answer the question from the following text: {question}? 
                                1. Find factual information from different media outlets
                                2. provide the evidence in the bullet point manner. 
                                3. Do not output anything else.""",
                    connectors=[{"id": "web-search"}]
                    )
                context.append(response.text)
        else:
            response = {'text': ''}

    return ' '.join(context)

def get_counter_claim(claim:str, qacontext:str):
    # co = cohere.Client(args.api_key)
    if not qacontext:
        counter = co.chat(
            model = 'command-r-plus',
            messages = [{
                    "role": "user",
                    "content": f"""Generate a succinct counter-argument that refutes the following claim: {claim}? 
                                   Provide only the answer and nothing else. 
                                   Do not give your opinion. 
                                   Make sure the answer is no longer than 3 sentences."""
                            }
                        ]   
                    )
    else:
        counter = co.chat(
                model = 'command-r-plus',
                messages = [{
                    "role": "user",
                    "content": f"""Given the following context: {qacontext}, provide  a succinct counter-argument that refutes the following argument using information from the context: {claim}. 
                                   Provide only the answer and nothing else. 
                                   Do not give your opinion. 
                                   Make sure the answer is no longer than 3 sentences.""",
                        }
                    ]
                )
    
    return counter.message.content[0].text, qacontext

# client = httpx.Client(timeout=None)
co = cohere.ClientV2(args.api_key)
rag(args.data_name)
