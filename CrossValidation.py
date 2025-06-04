import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import pickle

def cross_validate(model_class, dataset, k_folds, epochs, batch_size, device):
    
    def validation(model,data,device):
        val = model.eval()
        total = 0
        correct = 0
        for i, (img,label) in enumerate(data):
            with torch.no_grad():
                img = img.to(device)
                x = val(img)
                value, pred = torch.max(x,1)
                pred = pred.to('cpu')
                total += img.size(0)
                correct += torch.sum(pred == label)
                acc = float(correct * 100 /total)
        return acc
    
    results = {}
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{k_folds}")
        acc = []
        
        # Create data loaders for this fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Initialize model, loss, optimizer
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            # Training phase
            model.train()
            for batch in train_loader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    outputs = model(inputs)
                    val_loss += loss_fn(outputs, targets).item()
            
            Accuracy = validation(model,val_loader,device)
            acc.append(Accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(val_loader)}")
            print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {Accuracy}")


        # Save results for this fold
        results[fold] = acc
        # torch.save(model.state_dict(), f'model_fold_{fold}.pth')
        # pickle.dump(results, open('Folding.pickle', 'wb'))


    print(f"Cross-validation results: {results}")
    return results