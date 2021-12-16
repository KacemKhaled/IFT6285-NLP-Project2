# Code inspired from :
# https://huggingface.co/docs/transformers/perplexity#example-calculating-perplexity-with-gpt2-in-transformers
# Modified to fit our task

# Transformers installation
# ! pip install transformers datasets
# To install from source instead of the last release, comment the command above and uncomment the following one.
# ! pip install git+https://github.com/huggingface/transformers.git

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

class GPT2:
    def __init__(self):
        self.model,self.tokenizer, self.device = self.load_gpt2()


    def load_gpt2(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('GPU Available:', torch.cuda.is_available())
        model_id = 'gpt2'
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        return model, tokenizer, device

    # Downloading: 100% 666/666 [00:00<00:00, 15.1kB/s]
    # Downloading: 100% 3.02G/3.02G [01:40<00:00, 33.6MB/s]
    # Downloading: 100% 0.99M/0.99M [00:01<00:00, 1.34MB/s]
    # Downloading: 100% 446k/446k [00:00<00:00, 617kB/s]
    # Downloading: 100% 1.29M/1.29M [00:01<00:00, 1.34MB/s]


    def perplexity(self,sent):
        encodings = self.tokenizer(sent, return_tensors='pt')
        max_length = self.model.config.n_positions
        stride = 512

        nlls = []
        for i in range(0, encodings.input_ids.size(1), stride):
          begin_loc = max(i + stride - max_length, 0)
          end_loc = min(i + stride, encodings.input_ids.size(1))
          trg_len = end_loc - i    # may be different from stride on last loop
          input_ids = encodings.input_ids[:,begin_loc:end_loc].to(self.device)
          target_ids = input_ids.clone()
          target_ids[:,:-trg_len] = -100

          with torch.no_grad():
              outputs = self.model(input_ids, labels=target_ids)
              neg_log_likelihood = outputs[0] * trg_len

          nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return ppl.tolist()

# gpt2 = GPT2()
#
# ref = "why does everything have to become such a big issue ?"
# sent_1 = '? everything big why to become does have such issue a'
# sent_2 = "a big issue to have become such ? why does everything"
# sent_3 = "why does everything have to become such a big ? issue"
# sent_4 = "? why does everything have to become such a big issue"
#
# print(gpt2.perplexity(ref))
# print(gpt2.perplexity(sent_1))
# print(gpt2.perplexity(sent_2))
# print(gpt2.perplexity(sent_3))
# print(gpt2.perplexity(sent_4))