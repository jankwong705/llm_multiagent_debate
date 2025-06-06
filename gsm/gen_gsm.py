from openai import OpenAI

# OpenAI object with key and server url
client = OpenAI(api_key="EMPTY",  
        base_url="http://localhost:8000/v1")
import json
import numpy as np
import random
from tqdm import tqdm

MODEL = "Qwen/Qwen3-4B"

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    # Editted way of grabbing the conversation content 
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

if __name__ == "__main__":
    agents = 3
    rounds = 2
    random.seed(0)

    generated_description = {}
    # Open dataset here 
    questions = read_jsonl("test.jsonl")
    random.shuffle(questions)

    tokens_sent_received = []   # [(tokens_sent, tokens_received),...]

    for data in tqdm(questions[:100]):
        question = data['question']
        answer = data['answer']

        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2*round - 1)
                    agent_context.append(message)

                # Using VLLM model 
                completion = client.chat.completions.create(model=MODEL,
                    messages=agent_context,
                    n=1)
                
                tokens_sent = completion.usage.prompt_tokens
                tokens_received = completion.usage.completion_tokens

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

                tokens_sent_received.append((tokens_sent, tokens_received))

        generated_description[question] = (agent_contexts, answer)

    json.dump(generated_description, open("gsm_{}_{}.json".format(agents, rounds), "w"))
    print(tokens_sent_received)
    import pdb
    # pdb.set_trace()
    # print(answer)
    # print(agent_context)
