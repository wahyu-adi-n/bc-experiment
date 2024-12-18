from config import DEVICE
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision, \
                                        MulticlassRecall, MulticlassAUROC, MulticlassConfusionMatrix
from tqdm import tqdm                              
import os
import torch

device = DEVICE

def do_train(model, train_loader, criterion, optimizer, epoch, output_dir, best_metric,
            scheduler = None, val_loader = None, ckpt = None, resume = False):
    
    history = {
        'train_loss_savings' : [],
        'train_acc_savings' : [],
        'val_loss_savings'  : [],
        'val_acc_savings' : []
    }
    start_epoch = 0
    best_metric_value = -1
    # start ckpt not specified, try to load last ckpt
    if not ckpt and resume and os.path.exists(os.path.join(output_dir, 'ckpt', 'last.pth')):
        ckpt = os.path.join(output_dir, 'ckpt', 'last.pth')
        
    if ckpt:
        ckpt = torch.load(ckpt)
        assert ckpt['best_metric'] == best_metric, 'best metric mismatch'
        best_metric_value = ckpt['best_metric_value']
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'] if ckpt['scheduler_state_dict'] else None)
        start_epoch = ckpt['epoch'] + 1
        print(f'loaded ckpt from {os.path.join(output_dir, "ckpt", "last.pth")}, starting from epoch {start_epoch}')
    
    os.makedirs(os.path.join(output_dir, 'ckpt'), exist_ok=True)
    model.to(device)
    total_step = start_epoch * len(train_loader)

    writer = SummaryWriter(output_dir, flush_secs=10)

    for cur_epoch in range(start_epoch, epoch):
        
        pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader))
        pbar.desc = '[%s: epoch %2d, batch %3d] loss: %.5f' % \
                (output_dir, cur_epoch, 0, 0)

        model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for i, data in pbar:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            # calculate loss
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            # backpropagation
            loss.backward()
            # update parameters
            optimizer.step()

            running_loss = loss.item()
            
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

            pbar.desc = '[%s: epoch %2d, batch %3d] loss: %.5f' % \
                (output_dir, cur_epoch, i + 1, running_loss)

            total_step += train_loader.batch_size
            writer.add_scalar("train/Loss", running_loss, global_step = total_step)

        if scheduler:
            scheduler.step()
        pbar.close()
        
        train_accuracy = correct_predictions / total_predictions
        history['train_acc_savings'].append(train_accuracy)
        history['train_loss_savings'].append(running_loss / len(train_loader))
        writer.add_scalar("train/Accuracy", train_accuracy, global_step=cur_epoch)
        
        if val_loader:
            print("evaluating")
            eval_loss, _, _, _, metrics = do_eval(model, val_loader, loss_criterion = criterion)

            writer.add_scalar("val/Loss", eval_loss, global_step = cur_epoch)
            writer.add_scalar("val/F1", metrics['f1']['macro'], global_step = cur_epoch)
            writer.add_scalar("val/Accuracy", metrics['acc']['macro'], global_step = cur_epoch)
            writer.add_scalar("val/Precision", metrics['precision']['macro'], global_step = cur_epoch)
            writer.add_scalar("val/Recall", metrics['recall']['macro'], global_step = cur_epoch)
            writer.add_scalar("val/AUROC", metrics['auroc'], global_step = cur_epoch)

            if best_metric_value < metrics[best_metric]:
                torch.save({
                    'best_metric': best_metric,
                    'best_metric_value': metrics[best_metric],
                    'epoch': cur_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None}, os.path.join(output_dir, 'ckpt', 'best.pth'))
                best_metric_value = metrics[best_metric]
            
        torch.save({
                'best_metric': best_metric,
                'best_metric_value': metrics[best_metric],
                'epoch': cur_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None}, os.path.join(output_dir, 'ckpt', 'last.pth'))
        
        history['val_acc_savings'].append(metrics['acc']['macro'])
        history['val_loss_savings'].append(eval_loss)
        
    writer.close()
    return history


def do_eval(model, eval_loader, ckpt_path = None, loss_criterion = None):
    torch.cuda.empty_cache()

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])

    model.to(device)

    batch_count = 0
    loss = 0.0
    pred_all = torch.tensor([]).to(device)
    label_all = torch.tensor([]).to(device)
    output_all = torch.tensor([]).to(device)
    
    model.eval()
    with torch.no_grad():
        for data in tqdm(eval_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if loss_criterion:
                loss += loss_criterion(outputs, labels).item()
                batch_count += 1
            pred = outputs.argmax(dim=1) 
            pred_all = torch.cat((pred_all, pred), dim=0)
            label_all = torch.cat((label_all, labels), dim=0)
            output_all = torch.cat((output_all, outputs), dim=0)

    if loss_criterion and batch_count > 0:
        loss /= batch_count

    ma_f1_metric = MulticlassF1Score(model.num_classes, average='macro').to(device)
    mi_f1_metric = MulticlassF1Score(model.num_classes, average='micro').to(device)
    ma_p_metric = MulticlassPrecision(model.num_classes, average='macro').to(device)
    mi_p_metric = MulticlassPrecision(model.num_classes, average='micro').to(device)
    ma_r_metric = MulticlassRecall(model.num_classes, average='macro').to(device)
    mi_r_metric = MulticlassRecall(model.num_classes, average='micro').to(device)
    ma_acc_metric = MulticlassAccuracy(model.num_classes, average='macro').to(device)
    mi_acc_metric = MulticlassAccuracy(model.num_classes, average='micro').to(device)
    auroc_metric = MulticlassAUROC(model.num_classes, average='macro', thresholds=10).to(device)
    confusion_matrix_metric = MulticlassConfusionMatrix(model.num_classes).to(device)

    label_all = label_all.long()
    metrics = {}
    metrics['f1'] = {'macro': ma_f1_metric(pred_all, label_all).item(), 'micro': mi_f1_metric(pred_all, label_all).item()}
    metrics['acc'] = {'macro': ma_acc_metric(pred_all, label_all).item(), 'micro': mi_acc_metric(pred_all, label_all).item()}
    metrics['precision'] = {'macro': ma_p_metric(pred_all, label_all).item(), 'micro': mi_p_metric(pred_all, label_all).item()}
    metrics['recall'] = {'macro': ma_r_metric(pred_all, label_all).item(), 'micro': mi_r_metric(pred_all, label_all).item()}
    metrics['auroc'] = auroc_metric(output_all, label_all).item()
    metrics['confusion_matrix'] = confusion_matrix_metric(pred_all, label_all).cpu().numpy()
    
    return loss, label_all, pred_all, output_all, metrics
