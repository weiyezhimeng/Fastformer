from utils import test
import torch
import torch.nn as nn
from transformers import BertTokenizer
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--s1', type=int, default = None)
    parser.add_argument('--e1', type=int, default= None)
    args = parser.parse_args()
    file="../MIND/MINDlarge_test/behaviors.tsv"
    tokenizer_path =  f"../bert-mini"
    model_path_bert = f"../bert-mini"
    device='cpu:0'

    model_user = torch.load('./user.pth').to(device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, padding_side='left')                  
    model_news = torch.load('./news.pth').to(device)
    test(file,model_user,model_news,tokenizer,device,args.s1,args.e1)
