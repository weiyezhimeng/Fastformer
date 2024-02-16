from math import sqrt
from load_data import load_data
from train import train 
import torch
from transformers import BertModel,BertTokenizer
import torch.nn as nn
from config_ import config
epoch=10
batch=64
file_train="../MIND/MINDlarge_train/behaviors.tsv"
device='cpu:0'
tokenizer_path = model_path = f"../bert-mini"
lr=1e-5
tokenizer = BertTokenizer.from_pretrained(tokenizer_path, padding_side='left')   

#轮子
class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super(FastSelfAttention, self).__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.attention_head_size = int(config.hidden_size /config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim= config.hidden_size
        
        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
                
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5
        # add attention mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads*self.attention_head_size)
        pooled_query_repeat= pooled_query.repeat(1, seq_len,1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer=mixed_key_layer* pooled_query_repeat
        
        query_key_score=(self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
        
        # add attention mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        weighted_value =(pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer
      
        return weighted_value

class News_Encoder(nn.Module):
    def __init__(self, ):
        super(News_Encoder, self).__init__()
        self.embedding  = BertModel.from_pretrained(model_path).get_input_embeddings()
        self.multi_head = FastSelfAttention(config)
        self.news_layer = nn.Sequential(nn.Linear(256, 200),
                                        nn.Tanh(),  
                                        nn.Linear(200, 1),
                                        nn.Flatten(), nn.Softmax(dim=0)) 
    def forward(self, x):
        outputs = self.embedding(x)
        multi_attention=self.multi_head(outputs)
        attention_weight = self.news_layer(multi_attention).unsqueeze(2)
        new_emb = torch.sum(multi_attention * attention_weight, dim=1)  
        return new_emb

model_news = News_Encoder().to(device)

class User_Encoder(nn.Module):
    def __init__(self, ):
        super(User_Encoder, self).__init__()
        self.news_encoder = model_news
        self.multi_head = FastSelfAttention(config)
        self.news_layer = nn.Sequential(nn.Linear(256, 200),
                                        nn.Tanh(),  
                                        nn.Linear(200, 1),
                                        nn.Flatten(), nn.Softmax(dim=0)) 
    def forward(self, x):
        outputs = self.news_encoder(x).unsqueeze(0)
        multi_attention=self.multi_head(outputs)
        attention_weight = self.news_layer(multi_attention).unsqueeze(2)
        new_emb = torch.sum(multi_attention * attention_weight, dim=1)  
        return new_emb

model_user = User_Encoder().to(device)

for name, param in model_user.named_parameters():
    if name=="news_encoder.embedding.weight":
        param.requires_grad = False
    else:
        param.requires_grad = True

loader_train=load_data(file_train,batch)

train(tokenizer,model_user,model_news,device,lr,epoch,loader_train,batch)





