import os
import json
import torch
import argparse 

from datasets import load_dataset, Dataset
from transformers import (
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        HfArgumentParser,
        TrainingArguments,
        pipeline,
        logging,
)
import pandas as pd
from peft import LoraConfig, PeftModel
from tqdm import tqdm

if __name__ == '__main__':
        def run(data_name:str):
                arguments, counter_arguments = [], []
                if data_name == 'cmv':
                        res_file = f'cmv_{args.model_name}.txt'
                        data = [eval(l) for l in open('data/data.jsonl').readlines()]
                        for l in data:
                                arguments.append(l['argument'].strip())  
                                counter_arguments.append(l['counter-argument'].strip())
                
                if data_name =='conan':
                        print('LOADING DATA: ')
                        res_file = f'conan_{args.model_name}.txt'
                        data = load_dataset("HiTZ/CONAN-EUS", 'en')
                        data = pd.DataFrame(data['test']).drop_duplicates(['HS'])
                        arguments = data['HS'].to_list()
                        counter_arguments = data['CN'].to_list()


                if data_name == 'persuasive':
                        res_file = f'antropic_{args.model_name}.txt'
                        data = load_dataset("Anthropic/persuasion")
                        data = pd.DataFrame(data['train'])
                        arguments = data[data['rating_final'] == '1 - Strongly oppose'].drop_duplicates('claim')['claim'].to_list()
                        arguments += data[data['rating_final'] == '2 - Oppose'].drop_duplicates('claim')['claim'].to_list()
                        counter_arguments = data[data['rating_final'] == '1 - Strongly oppose'].drop_duplicates('claim')['argument'].to_list()
                        counter_arguments += data[data['rating_final'] == '2 - Oppose'].drop_duplicates('claim')['argument'].to_list()
                
                if data_name == 'opinion':
                        arguments = []
                        # on the first run

                        #dataset_name = 'Anthropic/llm_global_opinions'
                        #data = load_dataset(dataset_name)
                        #questions = data['train']['question']
                        #options = data['train']['options']
                        #assert len(questions) == len(options), "question and option lengths are not equal"
                        #print('Turning questions into opinions: ')
                        #for question, ops in tqdm(zip(questions[:50], options[:50]), total=len(questions[:50])):
                        #        ops = eval(ops)
                        #        for option in ops:
                        #                if isinstance(option, str) and 'DK/Refused' not in option:
                        #                         arguments.append(get_opinion(question, option))

                        # when we already have data - TODO: fix it 
                        data = [eval(l) for l in open('mistral-opinions-norag-qa.jsonl').readlines()]
                        arguments = []
                        for l in data:
                            arguments.append(l['argument'].strip())
                        print(f'{len(arguments)} claims/opinions generated.')
                        # TODO: remove necesity in counter-argument
                        counter_arguments = arguments


                output_file = open(args.output_file, 'w')
                
                for argument, counter_argument in tqdm(zip(arguments, counter_arguments), total=len(arguments)):

                        #questions = get_questions(argument)
                        #qacontext = get_qacontext(questions)

                        counter = get_counter_claim(argument, qacontext=None)
                        
                        output = {
                                'argument': argument, # input claim
                                'model': counter, # generated counter-claim 
                                #'questions': questions, # generated questions from the input claim
                                #'qa_context': qacontext, # answers to the generated questions using web-search 
                                # 'counter-argument': counter_argument # "gold standard" counter-claim
                        }
                        json.dump(output, output_file)
                        output_file.write('\n')

                      
                return 

        def get_opinion(question: str, option: str):
                
                        prompt = f'Merge two texts and make turn them into claim: {question} with position {option}? Provide answer from a first person perspective. Provide only one sentence. Do not generate anything else. Do not give your opinion. Answer: '
                        model_input = tokenizer(prompt, return_tensors='pt').to('cuda')
                        input = tokenizer(prompt, return_tensors='pt').to('cuda')
                        model_output = base_model.generate(**model_input, max_new_tokens=512, do_sample=True, top_k=50)
                        result = tokenizer.decode(model_output[0], skip_special_tokens=True)
                        result = result.split('Answer: ')[1]
                        # print(result)
                        # print()
                        return result

        def get_questions(claim: str):
                prompt = f'Generate 3 questions that will question the veracity of the given claim and persuade to take the opposing position: {claim}? Provide only querstions and nothing else. Questions: '
                model_input = tokenizer(prompt, return_tensors='pt').to('cuda')
                input = tokenizer(prompt, return_tensors='pt').to('cuda')
                model_output = base_model.generate(**model_input, max_new_tokens=512, do_sample=True, top_k=50)
                result = tokenizer.decode(model_output[0], skip_special_tokens=True)
                result = result.split('Questions: ')[1]

                return result

        def get_qacontext(questions: str):
        
                prompt = f'Answer each question from the follwoing text: {questions}? Provide short fact based answers and nothing else. Answer: '
                model_input = tokenizer(prompt, return_tensors='pt').to('cuda')
                input = tokenizer(prompt, return_tensors='pt').to('cuda')
                model_output = base_model.generate(**model_input, max_new_tokens=512, do_sample=True, top_k=50)
                result = tokenizer.decode(model_output[0], skip_special_tokens=True)
                response = result.split('Answer: ')[1]
                
                return response

        def get_counter_claim(claim: str, qacontext:str):
                if not qacontext:
                        prompt = f'Generate a succinct counter-argument that refutes the following claim: {claim}? Provide only the answer and nothing else. Do not give your opinion. Make sure the answer is no longer than 3 sentences. Counter-argument: '
                else:
                        prompt = f'Given the following context: {qacontext}, provide the counter-argument that refutes the following claim from the context: {claim}. Provide only the answer and nothing else. Do not give your opinion. Make sure the answer is no longer than 3 sentences. Counter-argument: '
                
                model_input = tokenizer(prompt, return_tensors='pt').to('cuda')
                input = tokenizer(prompt, return_tensors='pt').to('cuda')
                model_output = base_model.generate(**model_input, max_new_tokens=512, do_sample=True, top_k=50)
                result = tokenizer.decode(model_output[0], skip_special_tokens=True)
                counter = result.split('Counter-argument: ')[1]

                return counter
        

        parser = argparse.ArgumentParser()
        parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model from huggingface",
        )
        parser.add_argument(
            "--data",
            type=str,
            required=True,
            choices=['conan', 'cmv', 'persuasive', 'opinion'],
            default='conan',
            help="The data to use",
        )
        parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the generated data",
        )
        args = parser.parse_args()
        model_name = args.model_name
        base_model = AutoModelForCausalLM.from_pretrained(
        #base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map='auto',
                )

                                     
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        logging.set_verbosity(logging.CRITICAL)
        run(args.data)

        
