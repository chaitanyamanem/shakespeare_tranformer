import torch
import argparse
import tqdm
import os
import pickle



def generate(model, stoi, itos, max_new_tokens, prompt):
    idx = [stoi[char] for char in prompt]
    idx = torch.tensor(idx, dtype=torch.long).unsqueeze(dim=0).to("cuda") # adding batch dimesion


    gen_loop = tqdm.tqdm(total=max_new_tokens, desc="gen_progress")
    for _ in range(max_new_tokens):
        prompt = idx[:,-block_size:]
        pred = model(prompt)
        pred = pred[:,-1,:]
        probs = torch.softmax(pred, axis=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_idx], axis=1)
        gen_loop.update(1)

    generated_text = [itos[token.item()] for token in idx[0]]
    gen_loop.write("\n###############################")
    gen_loop.write("".join(generated_text))
    gen_loop.write("\n###############################")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path of the model")
    parser.add_argument("vocab_directory", help="directory where vocab files saved")
    parser.add_argument("max_new_tokens", help="Number of new tokens to generate by the model")
    parser.add_argument("prompt", help="prompt to begain the generation with")    
    parser.add_argument("-block_size", help="block size of the model", default=256, required=False)
    args = vars(parser.parse_args())
    model_path, max_new_tokens, prompt, block_size = args["model"], int(args["max_new_tokens"]), str(args["prompt"]), args["block_size"]
    vocab_directory = args["vocab_directory"]

    #Model and vocab files
    model = torch.load(model_path).to("cuda")    
    with open(os.path.join(vocab_directory,"stoi.pkl"), 'rb') as f:
        stoi = pickle.load(f)
    with open(os.path.join(vocab_directory,"itos.pkl"), 'rb') as f:
        itos = pickle.load(f)

    generate(model, stoi, itos, max_new_tokens, prompt)


