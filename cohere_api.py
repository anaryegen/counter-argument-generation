# from datasets import load_dataset
import glob
import pandas as pd
import time
import json
import argparse 
import random

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
            default='cmv',
            choices=['cmv', 'conan', 'persuasive', 'opinion'],
            help="Path to the data with claims to counter",
            # nargs='+'
        )
        parser.add_argument(
            "--data_path",
            type=str,
            default='data/data.jsonl',
            help="Path to the data with claims to counter",
        )
        # parser.add_argument(
        #     "--output_file",
        #     type=str,
        #     required=True,
        #     help="Path to save the generated data",
        # )
        args = parser.parse_args()

def rag(data_name: str):
    
    argument = []
    counter_argument = []
    if data_name == 'cmv':
        res_file = 'cmv_rag_qa.txt'
        data = [eval(l) for l in open(args.data_path).readlines()]
        for l in data:
            argument.append(l['argument'].strip())  
            counter_argument.append(l['counter-argument'].strip())
    
    if data_name =='conan':
        print('LOADING DATA: ')
        res_file = 'conan_save_data.txt'
        data = load_dataset("HiTZ/CONAN-EUS", 'en')
        data = pd.DataFrame(data['test']).drop_duplicates(['HS'])
        argument = data['HS'].to_list()
        counter_argument = data['CN'].to_list()


    if data_name == 'persuasive':
        res_file = 'antropic_save_data.txt'
        data = load_dataset("Anthropic/persuasion")
        data = pd.DataFrame(data['train'])
        argument = data[data['rating_final'] == '1 - Strongly oppose'].drop_duplicates('claim')['claim'].to_list()
        argument += data[data['rating_final'] == '2 - Oppose'].drop_duplicates('claim')['claim'].to_list()
        counter_argument = data[data['rating_final'] == '1 - Strongly oppose'].drop_duplicates('claim')['argument'].to_list()
        counter_argument += data[data['rating_final'] == '2 - Oppose'].drop_duplicates('claim')['argument'].to_list()
        argument = argument
        counter_argument = counter_argument

    if data_name == 'opinion':
        res_file = 'cmdr_opinion_rag_qa_128.txt'
        data = [eval(l) for l in open('mistral-opinions-norag-qa.jsonl').readlines()]
        argument = [] 
        for l in data[128:]:
            argument.append(l['argument'].strip())
        counter_argument = argument # temporal


    print("\tNumber of claims: " , len(argument))
    print("\tNumber of counter-claims: " , len(argument))



    count = 0
    results = open(res_file , 'w')
    for claim, counter_claim in tqdm(zip(argument, counter_argument), total=len(argument)):
        # count += 100
        #if count % 10 == 0:
            # to comply with the 10 API calls/min restriction
        #time.sleep(60)
        
        questions = get_questions(claim)
        qacontext = get_qacontext(questions, count)
        counter, qacontext = get_counter_claim(claim, qacontext) 

        output = {
                    'argument': claim, # input claim
                    'cmdr_websearch': counter, # generated counter-claim 
                    'questions': questions, # generated questions from the input claim
                    'qa_context': qacontext, # answers to the generated questions using web-search 
                    'counter-argument': counter_claim # "gold standard" counter-claim
                }      

        json.dump(output, results)
        results.write('\n')



def get_questions(claim: str):
    text = co.chat(
            model = 'command-r-plus',
            message = f'Generate 3 questions that will question the veracity of the given claim and persuade to take the opposing position: {claim}? Provide only questions and nothing else.',
        )
    return text.text

def get_qacontext(questions: str, counter: int):
    if not questions:
        print("Questions were not provided, make sure that the logic of the function call is correct :)")
    else:
        response = co.chat(
            model = 'command-r-plus',
            message = f'Answer each question from the following text: {questions}? Find factual information from different media outlets and provide the evidence in the bullet point manner. Do not output anything else.',
            connectors=[{"id": "web-search"}]
        )
        # time.sleep(60)

        # to save retireved external knowledge
        #for i in range(len(response.documents)):
        #    r = random.randrange(0, 100000000)
        #    wr = open(f'{args.data_name}_external_data/{r}.txt', 'w')
        #    wr.write(response.documents[i]['snippet'])
    return response.text

def get_counter_claim(claim:str, qacontext:str):
    
    if not qacontext:
        counter = co.chat(
            model = 'command-r-plus',
            message = f'Generate a succinct counter-argument that refutes the following claim: {claim}? Make sure the answer is not longer than 3 sentences.',
            #connectors=[{"id": "web-search"}]
        )
    else:
        counter = co.chat(
                model = 'command-r-plus',
                message = f'Given the following context: {qacontext}, \n\n Provide the counter-argument that refutes the following claim using information from the context: {claim}. \n\nProvide only the answer and nothing else. Do not give your opinion. Make sure the answer is no longer than 3 sentences.',
            )
    
    return counter.text, qacontext


co = cohere.Client(args.api_key)
rag(args.data_name)
