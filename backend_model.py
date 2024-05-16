import torch
import transformers
import numpy as np

from backend_utils import BBPETokenizerPPLCalc, SPLlamaTokenizerPPLCalc, CharLevelTokenizerPPLCalc, SPChatGLMTokenizerPPLCalc
from backend_utils import split_sentence
# mosec
from mosec import Worker
from mosec.mixin import MsgpackMixin
# llama
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


class SnifferBaseModel(MsgpackMixin, Worker):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = None
        self.base_model = None
        self.generate_len = 512

    def forward_calc_ppl(self):
        pass

    def forward_gen(self):
        self.base_tokenizer.padding_side = 'left'
        # 1. single generate
        if isinstance(self.text, str):
            tokenized = self.base_tokenizer(self.text, return_tensors="pt").to(
                self.device)
            tokenized = tokenized.input_ids
            gen_tokens = self.base_model.generate(tokenized,
                                                  do_sample=True,
                                                  max_length=self.generate_len)
            gen_tokens = gen_tokens.squeeze()
            result = self.base_tokenizer.decode(gen_tokens.tolist())
            return result
        # 2. batch generate
        # msgpack.unpackb(self.text, use_list=False) == tuple
        elif isinstance(self.text, tuple):
            inputs = self.base_tokenizer(self.text,
                                         padding=True,
                                         return_tensors="pt").to(self.device)
            gen_tokens = self.base_model.generate(**inputs,
                                                  do_sample=True,
                                                  max_length=self.generate_len)
            gen_texts = self.base_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            return gen_texts

    def forward(self, data):
        """
        :param data: ['text': str, "do_generate": bool]
        :return:
        """
        self.text = data["text"]
        self.do_generate = data["do_generate"]
        if self.do_generate:
            return self.forward_gen()
        else:
            return self.forward_calc_ppl()


class SnifferGPT2Model(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'gpt2-xl',cache_dir="seqXgpt_cache/")
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'gpt2-xl',cache_dir="seqXgpt_cache/")
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model.to(self.device)
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder,
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)
    
class SnifferGPT2FrenchModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'ClassCat/gpt2-base-french',cache_dir="seqXgpt_cache/")
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'ClassCat/gpt2-base-french',cache_dir="seqXgpt_cache/")
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model.to(self.device)
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder,
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)
    
class SnifferGPTNeoModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'EleutherAI/gpt-neo-2.7B',cache_dir="seqXgpt_cache/")
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'EleutherAI/gpt-neo-2.7B', device_map="auto", load_in_8bit=True,cache_dir="seqXgpt_cache/")
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder,
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)

class SnifferGPTNeoChineseModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'Langboat/mengzi-gpt-neo-base',cache_dir="seqXgpt_cache/")
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'Langboat/mengzi-gpt-neo-base', device_map="auto", load_in_8bit=True,cache_dir="seqXgpt_cache/")
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder,
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)


class SnifferGPTJModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'EleutherAI/gpt-j-6B',cache_dir="seqXgpt_cache/")
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'EleutherAI/gpt-j-6B', device_map="auto", load_in_8bit=True,cache_dir="seqXgpt_cache/")
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder,
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)
    
class SnifferGPTJFrenchModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'Cedille/fr-boris',cache_dir="seqXgpt_cache/")
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'Cedille/fr-boris', device_map="auto", load_in_8bit=True,cache_dir="seqXgpt_cache/")
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder,
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)
    
    
class SnifferLlamaModel(SnifferBaseModel):
    """
    More details can be seen:
        https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaModel
    """

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        model_path = 'huggyllama/llama-7b'
        self.base_tokenizer = LlamaTokenizer.from_pretrained(model_path,cache_dir="seqXgpt_cache/")
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_tokenizer.unk_token_id = self.base_tokenizer.unk_token_id
        self.base_model = LlamaForCausalLM.from_pretrained(model_path,
                                                           device_map="auto",
                                                           load_in_8bit=True,
                                                          cache_dir="seqXgpt_cache/")
        self.ppl_calculator = SPLlamaTokenizerPPLCalc(self.base_model,
                                                      self.base_tokenizer,
                                                      self.device)
    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)

class SnifferLlamaFrenchModel(SnifferBaseModel):
    """
    More details can be seen:
        https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaModel
    """

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        model_path = 'sinking8/llama2_basic_french'
        self.base_tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir="seqXgpt_cache/",trust_remote_code=True)
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_tokenizer.unk_token_id = self.base_tokenizer.unk_token_id
        self.base_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                           device_map="auto",
                                                           load_in_8bit=True,
                                                          cache_dir="seqXgpt_cache/",trust_remote_code=True)
        self.ppl_calculator = SPLlamaTokenizerPPLCalc(self.base_model,
                                                      self.base_tokenizer,
                                                      self.device)
        
    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)


class SnifferLlamaChineseModel(SnifferBaseModel):
    """
    More details can be seen:
        https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaModel
    """

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        model_path = 'hfl/chinese-llama-2-7b'
        self.base_tokenizer = LlamaTokenizer.from_pretrained(model_path,cache_dir="seqXgpt_cache/")
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_tokenizer.unk_token_id = self.base_tokenizer.unk_token_id
        self.base_model = LlamaForCausalLM.from_pretrained(model_path,
                                                           device_map="auto",
                                                           load_in_8bit=True,
                                                          cache_dir="seqXgpt_cache/")
        self.ppl_calculator = SPLlamaTokenizerPPLCalc(self.base_model,
                                                      self.base_tokenizer,
                                                      self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)
    
class SnifferLlama3Model(SnifferBaseModel):
    """
    More details can be seen:
        https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaModel
    """

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        model_path = 'BoyangZ/llama3_chinese_basemode_ftv1'
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path,cache_dir="seqXgpt_cache/")
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_tokenizer.unk_token_id = self.base_tokenizer.unk_token_id
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(model_path,
                                                           device_map="auto",
                                                           load_in_8bit=True,
                                                          cache_dir="seqXgpt_cache/")
        self.ppl_calculator = SPLlamaTokenizerPPLCalc(self.base_model,
                                                      self.base_tokenizer,
                                                      self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)

class SnifferWenZhongModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        # bpe tokenizer
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese',cache_dir="seqXgpt_cache/")
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese',
            device_map="auto",
            load_in_8bit=True,
            cache_dir="seqXgpt_cache/")
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder,
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)
    
class SnifferSkyWorkModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'SkyWork/SkyTextTiny', trust_remote_code=True,cache_dir="seqXgpt_cache/")
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'SkyWork/SkyTextTiny', device_map="auto", load_in_8bit=True,cache_dir="seqXgpt_cache/")
        all_special_tokens = self.base_tokenizer.all_special_tokens
        self.ppl_calculator = CharLevelTokenizerPPLCalc(
            all_special_tokens, self.base_model, self.base_tokenizer,
            self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)
    
class SnifferDaMoModel(SnifferBaseModel):

    def __init__(self):
        from modelscope.models.nlp import DistributedGPT3
        from modelscope.preprocessors import TextGenerationJiebaPreprocessor
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        model_dir = 'seqXgpt_cache/"'
        self.base_tokenizer = TextGenerationJiebaPreprocessor(model_dir)
        self.base_model = DistributedGPT3(model_dir=model_dir, rank=0)
        self.base_model.to(self.device)
        self.all_special_tokens = ['']

    def calc_sent_ppl(self, outputs, labels):
        lm_logits = outputs.logits.squeeze()  # seq-len, V
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        ll = loss_func(shift_logits, shift_labels.view(-1))  # [seq-len] ?
        loss = ll.mean().item()
        ll = ll.tolist()
        return loss, ll

    def calc_token_ppl(self, input_ids, ll):
        input_ids = input_ids[0].cpu().tolist()
        # char-level
        words = split_sentence(self.text)
        chars_to_words = []
        for idx, word in enumerate(words):
            char_list = list(word)
            chars_to_words.extend([idx for i in range(len(char_list))])

        # get char_level ll
        tokenized_tokens = []
        for input_id in input_ids:
            tokenized_tokens.append(self.base_tokenizer.decode([input_id]))
        chars_ll = []
        # because tokenizer don't include <s> before the sub_tokens,
        # so the first sub_token's ll cannot be obtained.
        token = tokenized_tokens[0]
        if token in self.all_special_tokens:
            char_list = [token]
        else:
            char_list = list(token)
        chars_ll.extend([0 for i in range(len(char_list))])
        # next we process the following sequence
        for idx, token in enumerate(tokenized_tokens[1:]):
            if token in self.all_special_tokens:
                char_list = [token]
            else:
                char_list = list(token)
            chars_ll.extend(ll[idx] for i in range(len(char_list)))

        # get token_level ll
        start = 0
        ll_tokens = []
        while start < len(chars_to_words) and start < len(chars_ll):
            end = start + 1
            while end < len(chars_to_words
                            ) and chars_to_words[end] == chars_to_words[start]:
                end += 1
            if end > len(chars_ll):
                break
            ll_token = chars_ll[start:end]
            ll_tokens.append(np.mean(ll_token))
            start = end

        # get begin_word_idx
        begin_token = self.base_tokenizer.decode([input_ids[0]])
        if begin_token in self.all_special_tokens:
            char_list = [begin_token]
        else:
            char_list = list(begin_token)
        begin_word_idx = chars_to_words[len(char_list) - 1] + 1

        return ll_tokens, begin_word_idx

    def forward_calc_ppl(self):
        # bugfix: clear the self.inference_params, so we can both generate and calc ppl
        self.base_model.train()
        self.base_model.eval()
        input_ids = self.base_tokenizer(self.text)['input_ids'].to(self.device)
        labels = input_ids
        input_ids = input_ids[:, :1024, ]
        labels = labels[:, :1024, ]
        outputs = self.base_model(tokens=input_ids,
                                  labels=input_ids,
                                  prompts_len=torch.tensor([input_ids.size(1)
                                                            ]))

        loss, ll = self.calc_sent_ppl(outputs, labels)
        ll_tokens, begin_word_idx = self.calc_token_ppl(input_ids, ll)
        return [loss, begin_word_idx, ll_tokens]

    def forward_gen(self):
        # 1. single generate
        if isinstance(self.text, str):
            input_ids = self.base_tokenizer(self.text)['input_ids'].to(
                self.device)
            gen_tokens = self.base_model.generate(input_ids,
                                                  do_sample=True,
                                                  max_length=self.generate_len)
            gen_tokens = gen_tokens.sequences
            gen_tokens = gen_tokens[0].cpu().numpy().tolist()
            result = self.base_tokenizer.decode(gen_tokens)
            return result
        # 2. batch generate
        # damo model didn't implement batch_encode and batch_decode, so we use a for loop here
        elif isinstance(self.text, tuple):
            batch_res = []
            for text in self.text:
                input_ids = self.base_tokenizer(text)['input_ids'].to(
                    self.device)
                gen_tokens = self.base_model.generate(
                    input_ids, do_sample=True, max_length=self.generate_len)
                gen_tokens = gen_tokens.sequences
                gen_tokens = gen_tokens[0].cpu().numpy().tolist()
                result = self.base_tokenizer.decode(gen_tokens)
                batch_res.append(result)
            return batch_res

class SnifferChatGLMModel(SnifferBaseModel):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "THUDM/chatglm-6b", trust_remote_code=True,cache_dir="seqXgpt_cache/")
        self.base_model = transformers.AutoModel.from_pretrained(
            "THUDM/chatglm-6b",
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=True,
            cache_dir="seqXgpt_cache/")
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.ppl_calculator = SPChatGLMTokenizerPPLCalc(
            self.base_model, self.base_tokenizer, self.device)

    def forward_calc_ppl(self):
        # self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)

    def forward_gen(self):
        self.base_tokenizer.padding_side = 'left'
        # 1. single generate
        if isinstance(self.text, str):
            inputs = self.base_tokenizer(self.text,
                                         padding=True,
                                         return_tensors="pt").to(self.device)
            gen_tokens = self.base_model.generate(**inputs,
                                                  do_sample=True,
                                                  max_new_tokens=self.generate_len)
            gen_texts = self.base_tokenizer.batch_decode(
                gen_tokens.tolist(), skip_special_tokens=True)
            result = gen_texts[0]
            return result
        # 2. batch generate
        # msgpack.unpackb(self.text, use_list=False) == tuple
        elif isinstance(self.text, tuple):
            self.text = list(self.text)
            inputs = self.base_tokenizer(self.text,
                                         padding=True,
                                         return_tensors="pt").to(self.device)
            gen_tokens = self.base_model.generate(
                **inputs, do_sample=True, max_new_tokens=self.generate_len)
            gen_texts = self.base_tokenizer.batch_decode(
                gen_tokens.tolist(), skip_special_tokens=True)
            return gen_texts
        

    
