from model import Model
from tokenizer import Tokenizer
import torch

weight_file = "model.pt"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

m = Model().to(device)
m.load_state_dict(torch.load(weight_file, weights_only=True))
t = Tokenizer()

prompt = "i think it"
def generate():
    prompt_tokens = t.tokenize(prompt, False)
    for _ in range(10):
        input = torch.tensor([prompt_tokens]).to(device)
        logits = m(input)
        index = torch.argmax(logits[0, -1, :], dim =-1)
        prompt_tokens.append(index)
    print(t.decode(prompt_tokens))

generate()