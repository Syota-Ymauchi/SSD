import torch
from tqdm import tqdm
from LossModule import SSDLoss

def learn(model, num_epochs, optimizer, train_loader, val_loader, num_classes=21, save_path=None, early_stop=False, device='cpu'):

    model.to(device)
    criterion = SSDLoss(num_classes, model.priors, device=device)
    # early stop
    best_total_val_loss = float('inf')
    no_update = 0
    for epoch in range(num_epochs):
        train_total_losses = []
        val_total_losses = []

        train_loc_losses = []
        val_loc_losses = []

        train_cls_losses = []
        val_cls_losses = []

        running_train_total_losses = 0.0
        running_val_total_losses = 0.0

        running_train_loc_losses = 0.0
        running_val_loc_losses = 0.0

        running_train_cls_losses = 0.0
        running_val_cls_losses = 0.0

        model.train()
        for imgs, anotations in tqdm(train_loader, desc='now training', total=len(train_loader), leave=False):
            
            imgs = imgs.to(device)
            anotations = anotations

            optimizer.zero_grad()
            lout, cout, priors = model.forward(imgs)
            # import pdb; pdb.set_trace()

            total_loss, cls_loss, loc_loss = criterion(lout, cout, anotations)

            total_loss.backward()
            optimizer.step()

            running_train_total_losses += total_loss.item()
            running_train_loc_losses += cls_loss.item()
            running_train_cls_losses += loc_loss.item()

        model.eval()
        for val_imgs, val_anotations in tqdm(val_loader, desc='now validation', total=len(val_loader), leave=False):

            val_imgs = val_imgs.to(device)
            val_anotations = val_anotations

            val_lout, val_cout, val_priors = model.forward(val_imgs)
            val_total_loss, val_cls_loss, val_loc_loss = criterion(val_lout, val_cout, val_anotations)

            running_val_total_losses += val_total_loss.item()
            running_val_loc_losses += val_cls_loss.item()
            running_val_cls_losses += val_loc_loss.item()

            



            

        train_total_losses.append(running_train_total_losses / len(train_loader))
        val_total_losses.append(running_val_total_losses / len(val_loader))
        train_loc_losses.append(running_train_loc_losses / len(train_loader))
        val_loc_losses.append(running_val_loc_losses / len(val_loader))
        train_cls_losses.append(running_train_cls_losses / len(train_loader))
        val_cls_losses.append(running_val_cls_losses / len(val_loader))

        if val_total_losses[-1] < best_total_val_loss:
                best_total_val_loss = val_total_losses[-1]
                no_update = 0
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)

        else:
            no_update +=1
            if early_stop  and early_stop <= no_update:
                break
        print(f"epoch {epoch+1}: train total loss {train_total_losses[-1]:.4f}, val total loss {val_total_losses[-1]:.4f}")


    return train_total_losses, val_total_losses, train_loc_losses, val_loc_losses, train_cls_losses, val_cls_losses