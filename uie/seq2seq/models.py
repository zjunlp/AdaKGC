import torch
from torch import nn
from .modeling_t5 import T5ForConditionalGeneration
import logging
logger = logging.getLogger("__main__")


class T5Prompt(nn.Module):
    def __init__(self, tokenizer, model_name_or_path, args):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        '''修改token embeddings大小'''
        self.t5.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model tokenizer length: {len(tokenizer)}")

        self.config = self.t5.config
        self.match_n_layer = self.config.num_decoder_layers
        self.match_n_head = self.config.num_heads
        self.n_embd = self.config.d_model
        self.match_n_embd = self.config.d_kv
        
        '''
        prompt参数
        use_prompt是否使用prompt
        '''
        self.prompt_len = args.prompt_len
        self.prompt_dim = args.prompt_dim
        self.prompt_inputs = torch.zeros(self.prompt_len).long()    # [0, 1, ..., prompt_len-1]
        self.use_prompt = args.use_prompt  
        self.map = {}

        self.wte = nn.Embedding(self.prompt_len, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.prompt_dim),
            nn.Tanh(),
            nn.Linear(self.prompt_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        
        self.wte_enc = nn.Embedding(self.prompt_len, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.prompt_dim),
            nn.Tanh(),
            nn.Linear(self.prompt_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        
        self.wte_dec = nn.Embedding(self.prompt_len, self.n_embd)
        self.control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.prompt_dim),
            nn.Tanh(),
            nn.Linear(self.prompt_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        self.dropout = nn.Dropout(0.1)


    
    def get_ids(self, ids):
        '''
        获得该ids对应的词向量的平均值
        '''
        ids = torch.tensor(ids, dtype=torch.long, device=self.t5.device)
        if ids.size(0) == 1:
            return self.t5.shared(ids)
        else:
            return torch.mean(self.t5.shared(ids), 0)


    def get_prompt_ids(self, ids):
        if ids in self.map.keys():
            index = torch.tensor(self.map[ids], dtype=torch.long, device=self.t5.device)
        else:
            embed = self.get_ids(ids).squeeze(0)
            logit = self.wte(torch.arange(self.prompt_len, dtype=torch.long, device=self.t5.device)).matmul(embed)
            index = torch.topk(torch.log_softmax(logit, dim = 0), k=1)[1]
        return index
        

    
    def init_prompt(self, spot_ids, asoc_ids, negative_sample, spot_prompt_id, asoc_prompt_id, pad_id):
        token_ids_embedding = torch.randn((self.prompt_len, self.n_embd), dtype = torch.float32, device=self.t5.device)
        '''
        用ids对应的词向量初始化prompt的值
        '''
        token_ids_embedding[0] = self.get_ids(pad_id)
        self.map[tuple(pad_id)] = 0
        token_ids_embedding[3] = self.get_ids(spot_prompt_id)
        self.map[tuple(spot_prompt_id)] = 3
        token_ids_embedding[4] = self.get_ids(asoc_prompt_id)
        self.map[tuple(asoc_prompt_id)] = 4
        logger.info(f"Before token_ids_embedding: {token_ids_embedding}")
        logger.info(f"Before wte: {self.wte(self.prompt_inputs)}")
        logger.info(f"Before wte_enc: {self.wte_enc(self.prompt_inputs)}")
        logger.info(f"Before wte_dec: {self.wte_dec(self.prompt_inputs)}")

        count = 5
        for spot in spot_ids:
            self.map[tuple(spot)] = count
            token_ids_embedding[count] = self.get_ids(spot)
            count += 1

        for asoc in asoc_ids:
            self.map[tuple(asoc)] = count
            token_ids_embedding[count] = self.get_ids(asoc)
            count += 1

        self.start = count
        for it in negative_sample:
            token_ids_embedding[count] = self.get_ids(it)
            count += 1

        self.wte = self.wte.from_pretrained(token_ids_embedding)
        self.wte_enc = self.wte_enc.from_pretrained(token_ids_embedding)
        self.wte_dec = self.wte_dec.from_pretrained(token_ids_embedding)

        logger.info(f"After token_ids_embedding: {token_ids_embedding}")
        logger.info(f"After wte: {self.wte(self.prompt_inputs)}")
        logger.info(f"After wte_enc: {self.wte_enc(self.prompt_inputs)}")
        logger.info(f"After wte_dec: {self.wte_dec(self.prompt_inputs)}")

    

    def get_input_tokens(self, bsz, spots, asocs):
        input_tokens = self.prompt_inputs.unsqueeze(0).expand(bsz, -1).clone().to(self.t5.device)
        spots_len = len(spots[0])

        input_tokens[:, 0] = 1
        input_tokens[:, 1] = 2
        input_tokens[:, 2] = 3
        start_pos = 3
        for i, spot in enumerate(spots):
            for j, s in enumerate(spot, start = start_pos):
                input_tokens[i, j] = self.get_prompt_ids(tuple(s))
        input_tokens[:, start_pos + spots_len] = 3

        input_tokens[:, start_pos + 1 + spots_len] = 4
        start_pos = 5 + spots_len
        for i, asoc in enumerate(asocs):
            for j, a in enumerate(asoc, start = start_pos):
                input_tokens[i, j] = self.get_prompt_ids(tuple(a))
        input_tokens[:, 5 + spots_len + len(asocs[0])] = 4

        return input_tokens

       
    def get_prompt(self, bsz, spots, asocs):
        input_tokens = self.get_input_tokens(bsz, spots, asocs)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        past_key_values_dec = self.control_trans_dec(
            temp_control_dec
        )  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        temp_control_enc = self.wte_enc(input_tokens)
        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": input_tokens.eq(0)
                    .to(key_val.device)
                    .bool()
                # bsz, prompt_len
            }
            key_val_dec = past_key_values_dec[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": input_tokens.eq(0)
                    .to(key_val_dec.device)
                    .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": input_tokens.eq(0)
                    .to(key_val_enc.device)
                    .bool(),
            }
            result.append(temp)

        return result
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        return_dict=True,
        spot=None,
        asoc=None,
    ):  
        '''
        使用use_prompt才会self.get_prompt获取prompt
        '''
        bs = input_ids.size(0)
        if self.use_prompt:
            past_prompt = self.get_prompt(bs, spot, asoc)
        else:
            past_prompt = None
        return self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            return_dict=return_dict,
            past_prompt=past_prompt,
        )
    

    def generate(
        self,
        input_ids,
        attention_mask,
        spot,
        asoc,
        **kwargs,
    ):
        bsz = input_ids.shape[0]
        past_prompt = None
        if self.use_prompt:
            past_prompt = self.get_prompt(bsz=bsz, spots=spot, asocs=asoc)
        else:
            past_prompt = None
        
        generated_ids = self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids