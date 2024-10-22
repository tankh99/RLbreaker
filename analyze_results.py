import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import openai
import parser
import argparse
import torch
import pandas as pd
from utils import check_keywords, check_with_unaligned, deepinfra_request, openai_request, test_prefixes, judge_prompt
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='post analysis')
    parser.add_argument('--file_path', type=str, default=None, help='path to model response')
    parser.add_argument('--target_model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--openai_key', type=str, default='You must have an OpenAI key', help='OpenAI key')
    parser.add_argument('--deepinfra_key', type=str, default='You must have an Deepinfra key', help='Deepinfra key')
    parser.add_argument('--cuda_id',type=int, default=0)
    args = parser.parse_args()
    openai.api_key = args.openai_key
    
    device = torch.device("cuda:{}".format(args.cuda_id))
    responses = pd.read_csv(args.file_path)['response'].tolist()
    all_questions = pd.read_csv(args.file_path)['question'].tolist()
    unalign_responses = pd.read_csv("datasets/question_unalign_res.csv")['response'].tolist()
    unalign_responses_q = pd.read_csv("datasets/question_unalign_res.csv")['question'].tolist()
    embedder = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
    # keyword / cosine similarity / gpt-judge
    keyword_asr = []
    similarities = []
    gpt_judge = []

    for i, q in enumerate(all_questions):
        print(i, q)
        cur_response = responses[i]
        success = check_keywords(cur_response, test_prefixes)
        keyword_asr.append(success)
        
        q_idx = unalign_responses_q.index(q)
        cur_unalign_res = unalign_responses[q_idx]
        similarity = check_with_unaligned(embedder, cur_response, cur_unalign_res)
        similarities.append(similarity)
        
        usr_prompt = judge_prompt.format(question=q, response=cur_response)
        # gpt_judge_result_text = openai_request(usr_prompt, model='gpt-4o-mini').choices[0].message.content
        gpt_judge_result_text = deepinfra_request(usr_prompt, model='microsoft/WizardLM-2-8x22B', api_key=args.deepinfra_key).choices[0].message.content
        gpt_judge_result = ('True' in gpt_judge_result_text or '1' in gpt_judge_result_text)
        gpt_judge.append(gpt_judge_result)
       
    print(f'{args.target_model} Key Word ASR: {sum(keyword_asr)/len(keyword_asr)}\n')
    print(f'{args.target_model} Similarity: {sum(similarities)/len(similarities)}\n')
    print(f'{args.target_model} GPT-Judge: {sum(gpt_judge)/len(gpt_judge)}\n')
    