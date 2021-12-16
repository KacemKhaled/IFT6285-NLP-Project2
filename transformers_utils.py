# Source of the code :
# https://huggingface.co/docs/transformers/perplexity#example-calculating-perplexity-with-gpt2-in-transformers
# Modified slightly to fit our task

# Transformers installation
# ! pip install transformers datasets
# To install from source instead of the last release, comment the command above and uncomment the following one.
# ! pip install git+https://github.com/huggingface/transformers.git

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

def load_gpt2():
    device = 'cuda'
    model_id = 'gpt2-large'
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    return model, tokenizer

# Downloading: 100% 666/666 [00:00<00:00, 15.1kB/s]
# Downloading: 100% 3.02G/3.02G [01:40<00:00, 33.6MB/s]
# Downloading: 100% 0.99M/0.99M [00:01<00:00, 1.34MB/s]
# Downloading: 100% 446k/446k [00:00<00:00, 617kB/s]
# Downloading: 100% 1.29M/1.29M [00:01<00:00, 1.34MB/s]

# # No need for this part
# from datasets import load_dataset
# test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
# encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

def perplexity_gpt2(sent):
  encodings = tokenizer(sent, return_tensors='pt')
  max_length = model.config.n_positions
  stride = 512

  nlls = []
  for i in range(0, encodings.input_ids.size(1), stride):
      begin_loc = max(i + stride - max_length, 0)
      end_loc = min(i + stride, encodings.input_ids.size(1))
      trg_len = end_loc - i    # may be different from stride on last loop
      input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
      target_ids = input_ids.clone()
      target_ids[:,:-trg_len] = -100

      with torch.no_grad():
          outputs = model(input_ids, labels=target_ids)
          neg_log_likelihood = outputs[0] * trg_len

      nlls.append(neg_log_likelihood)

  ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
  return ppl.tolist()