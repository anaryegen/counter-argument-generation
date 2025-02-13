import json
import argparse 

from tqdm import tqdm
import cohere


if __name__ == '__main__':
        
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--api_key",
            type=str,
            required=True,
            help="API key for Cohere",
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

def run(data_name: str):

    res_file = args.output_file
    argument = open("data/argument.txt").readlines()
    counter_argument = open("data/counter-argument.txt").readlines()

    print("\tNumber of claims: " , len(argument))
    print("\tNumber of counter-claims: " , len(counter_argument))
    assert len(argument) == len(counter_argument), "The length of argument and counter-argument are different. Please check your data."

    output_file = open(res_file , 'w')
    for claim, counter_claim in tqdm(zip(argument, counter_argument), total=len(argument)):

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
        
        json.dump(output, output_file, indent=4)
        # output_file.write('\n')


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

co = cohere.ClientV2(args.api_key)
run(args.data_name)
