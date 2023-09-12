import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class DialoBot():
    model_url = "tinkoff-ai/ruDialoGPT-medium"
    my_model_path = 'output'

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_url)
        self.model = AutoModelForCausalLM.from_pretrained(self.my_model_path).cpu()
        self.device = torch.device('cpu')

    def __call__(self, messages: list) -> str:
        all_text = ''
        for step, input_user in enumerate(messages):
            if step % 2 == 0:
                all_text += "@@ПЕРВЫЙ@@" + input_user
            else:
                if step != len(messages) - 1:
                    all_text += "@@ВТОРОЙ@@" + input_user
                else:
                    all_text += "@@ВТОРОЙ@@" + input_user + "@@ПЕРВЫЙ@@"

        inputs = self.tokenizer(all_text, return_tensors='pt').to(self.device)
        output = self.model.generate(
            **inputs,
            top_k=10,
            top_p=0.95,
            num_beams=3,
            num_return_sequences=3,
            do_sample=True,
            no_repeat_ngram_size=2,
            temperature=1.2,
            repetition_penalty=1.2,
            length_penalty=1.0,
            eos_token_id=50257,
            max_new_tokens=40
        )
        context_with_response = [self.tokenizer.decode(sample_token_ids) for sample_token_ids in output]
        l = len(all_text.split('@@'))
        out = context_with_response[0].split('@@')[l - 1]
        return out