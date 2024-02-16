import torch
from utils import loss
import gc
from GPU import GPU
from tqdm import tqdm

def train(tokenizer,model_user,model_news,device,lr,EPOCH,loader,batch):
    # ========== setup optim_embeds ========== #
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model_user.parameters()),lr=lr,weight_decay=0,eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20000)
    for epoch in range(EPOCH):
        print('Epoch:', epoch + 1, 'Training...')
        for step, (history, label) in tqdm(enumerate(loader)):  # 每一步loader释放一小批数据用来学习
            # ========== setup optimizer and scheduler ========== #
            loss_all=loss(history,label,batch,model_user,model_news,tokenizer,device)
            print("loss:", loss_all, "step:", step)
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            scheduler.step()
            if step==20000:
                break
            del  loss_all ; gc.collect();torch.cuda.empty_cache()
    torch.save(model_user, 'user.pth')
    torch.save(model_news, 'news.pth')







