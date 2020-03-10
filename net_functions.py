import numpy as np
import pandas as pd
import torch


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """
    Trains a model and saves the one with the best validation loss
    Params
        n_epochs: number of epochs to train the model
        loaders: pytorch DataLoader instances dict (should contain 'train' and 'valid' keys)
        model: pytorch model to train
        optimizer: pytorch optim
        criterion: pytorch criterion
        use_cuda: whether to use CUDA GPU accelerator
        save_path: path to save the model when validation loss improves
    """
    # Initialization for validation loss tracking
    valid_loss_min = np.Inf
    last_valid_loss = 0
    
    for epoch in range(1, n_epochs+1):
        # Variables for loss monitoring
        train_loss = 0.0
        valid_loss = 0.0
        
        # Training
        model.train()
        i = 0
        for batch_idx, (data, target, *_) in enumerate(loaders['train']):
            i += 1
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                
            data, target = data.contiguous(), target.contiguous()
                
            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores, target)
            loss.backward()
            optimizer.step()
            
            train_loss += ((1 / (batch_idx+1)) * (loss.item() - train_loss))
            
        # Validation
        model.eval()
        i = 0
        for batch_idx, (data, target, *_) in enumerate(loaders['valid']):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                
            data, target = data.contiguous(), target.contiguous()
                
            scores = model(data)
            loss = criterion(scores, target)
            valid_loss += ((1 / (batch_idx+1)) * (loss.item() - valid_loss))

        last_valid_loss = valid_loss
        
        # Save the model if validation loss has decreased
        if valid_loss < valid_loss_min and epoch > 1:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f} \tCheckpoint!')
        else:
            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')
            
    return
    
    
def test_to_csv(loaders, MM_model, SK_model, use_cuda):
    
    Ids = []
    task_1 = []
    task_2 = []
    
    MM_model.eval()
#     SK_model.eval()
    for batch_idx, (data, target, path) in enumerate(loaders['test']):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        
        Ids.append(path)
        
        output = MM_model(data)
        pred = output.data.max(1, keepdim=True)[1]
#         _, pred = torch.max(output, 1)
        task_1.append(pred.data.cpu().numpy())
        
#         output = SK_model(data)
#         pred = output.data.max(1, keepdim=True)[1]
#         task_2.append(pred.data.cpu().numpy())

    df = pd.DataFrame({
        'Id': np.array(Ids).flatten(),
        'task_1': np.array(task_1).flatten(),
        'task_2': np.zeros_like(np.array(task_1).flatten()),
#         'task_2': np.array(task_2).flatten(),
    }, columns= ['Id', 'task_1', 'task_2'])
    
    df.to_csv('predictions.csv', index=False)
    print('predictions.csv saved!')
    
    return df