from .Embedding import BertEmbeddings
from .Encoder import BertEncoder
from .Pooler import BertPooler

import torch
from torch import nn
from .ForPretrainedModel import BertPreTrainedModel
def get_extended_attention_mask(attention_mask, input_shape, device = None, dtype = torch.float32):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
        
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask

class BertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.post_init()
        
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        save_hidden = False
        
    ):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        
        past_key_values_length = 0
        extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape)
        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            past_key_values_length=past_key_values_length,
        )
        ##########################################
        #save_cls means [CLS] tokens of all layer#
        ##########################################
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            save_hidden = save_hidden
        )
        pooled_output = self.pooler(encoder_outputs[0])
        
        return (embedding_output, encoder_outputs[0], pooled_output, encoder_outputs[1])