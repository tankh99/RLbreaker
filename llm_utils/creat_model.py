import torch
from fastchat.model import load_model


@torch.inference_mode()
def create_model(args, model_path):
    model, tokenizer = load_model(
        model_path,
        args.device,
        args.num_gpus,
        args.dtype,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )
    return model, tokenizer


def create_model_and_tok(args, model_path, target=False):
    openai_model_list = ['gpt-4o-mini', 'gpt-4o-2024-05-13','meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Llama-2-70b-chat-hf','mistralai/Mixtral-8x7B-Instruct-v0.1', 'mistralai/Mixtral-8x22B-Instruct-v0.1', 'gpt-3.5-turbo-1106','gpt-3.5-turbo-0613', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301']
   
    if model_path in openai_model_list:
        MODEL = model_path
        TOK = None
    else:
        raise NotImplementedError

    return MODEL, TOK


def prepare_model_and_tok(args, target=False):
    if type(args.model_path) == str:
        MODEL, TOK = create_model_and_tok(args, args.model_path, target=target)
    elif type(args.model_path) == list:
        MODEL, TOK = [], []
        for model_path in args.model_path:
            model, tok = create_model_and_tok(args, model_path)
            MODEL.append(model)
            TOK.append(tok)
    else:
        raise NotImplementedError

    return MODEL, TOK