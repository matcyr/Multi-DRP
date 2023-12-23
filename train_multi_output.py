import torch
import torch.nn as nn
import sys
sys.path.append('/home/yurui/GDSC_2/code')
from code.model.drp_model.DRP_multi_conc import *
from code.loader.GDSC2_loader import *
import warnings
import argparse
import datetime
import torch.cuda.amp as amp
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore", category=UserWarning)
def arg_parse():
    parser = argparse.ArgumentParser(description="Model Configuration")
    parser.add_argument('--drug_embed_dim', type=int, default=128,
                        help='Embedding dimension for drug')
    parser.add_argument('--cell_embed_dim', type=int, default=128,
                        help='Embedding dimension for cell')
    parser.add_argument('--drug_layer_num', type=int, default=2,
                        help='Number of layers for drug')
    parser.add_argument('--cell_layer_num', type=int, default=2,
                        help='Number of layers for cell')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--readout', type=str, default='mean',
                        help='Readout function')
    parser.add_argument('--JK', type=str, default='True',
                        help='JKNet option')
    parser.add_argument('--view_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)    
    parser.add_argument('--lr', type= float, default = 1e-3,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    # parser.add_argument('--device', type = str, default = 0,
    #                     help='Device')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                        help='Weight decay')
    parser.add_argument('--check_step', type=int, default = 5,
                        help='Num of steps to check performance')
    parser.add_argument('--use_regulizer', type=str, default='True')
    parser.add_argument('--regular_weight', type=float, default= 1.0)
    parser.add_argument('--use_drug_path_way', type=str, default='True')
    parser.add_argument('--regular_weight_drug_path_way', type=float, default= 1.0)
    parser.add_argument('--train_type', type=str, default='Mixed')
    parser.add_argument('--scheduler_type', type=str, default='OP')
    parser.add_argument('--device',type = int, default= 0)
    parser.add_argument('--early_stop_count',type = int, default= 7)
    return parser.parse_args()
def cross_entropy_loss(input, target):
    return F.cross_entropy(input, target)
def total_loss(out_dict, batch_sample, args):
    m = args.regular_weight
    p = args.regular_weight_drug_path_way
    pred_loss = nn.MSELoss()(out_dict['pred'], batch_sample['label'])
    class_l = 0.0
    drug_pathway_l = 0.0
    if args.use_regulizer == 'True':
        class_l = cross_entropy_loss(out_dict['cell_regulizer'], batch_sample['CL_type'])
    if args.use_drug_path_way == 'True':
        drug_pathway_l = cross_entropy_loss(out_dict['drug_pathway'], batch_sample['drug_atom_repr'].PATHWAY_TYPE)
    return pred_loss + m * class_l + p*drug_pathway_l
def train_step(model, train_loader, optimizer, writer, epoch, device, args):
    # enable automatic mixed precision
    scaler = amp.GradScaler()

    model.train()
    y_true, preds = [], []
    optimizer.zero_grad()
    for data in tqdm(train_loader):
        batch_sample = {k: v.to(device) for k, v in data.items()}              
        with amp.autocast():
            out_dict = model(batch_sample)
            loss = total_loss(out_dict, batch_sample, args)
        preds.append(out_dict['pred'].float())
        y_true.append(batch_sample['label'])
        # perform backward pass and optimizer step using the scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # scheduler.step()
    y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(preds, dim=0).cpu().detach().numpy()
    rmse_dict = {}
    pcc_dict = {}
    r2_dict = {}
    MAE_dict = {}
    for i in range(7):
        column_true = y_true[:, i]
        column_pred = y_pred[:, i]
        rmse = mean_squared_error(column_true, column_pred, squared=False)
        rmse_dict[i] = rmse
        pcc = pearsonr(column_true, column_pred)[0]
        pcc_dict[i] = pcc
        r_2 = r2_score(column_true, column_pred)
        r2_dict[i] = r_2
        MAE = mean_absolute_error(column_true, column_pred)
        MAE_dict[i] = MAE
        print(f'Train accuracy for dose {i}: RMSE: {rmse:.4f}, PCC: {pcc:.4f}, R2: {r_2:.4f}, MAE: {MAE:.4f}')
        writer.add_scalar("Loss", rmse, epoch)
        writer.add_scalar(f"Accuracy/train/response_for_conc_{i+1}/rmse", rmse, epoch)
        writer.add_scalar(f"Accuracy/train/response_for_conc_{i+1}/mae", MAE, epoch)
        writer.add_scalar(f"Accuracy/train/response_for_conc_{i+1}/pcc", pcc, epoch)
        writer.add_scalar(f"Accuracy/train/response_for_conc_{i+1}/r_2", r_2, epoch)
    print(optimizer.param_groups[0]['lr'])
    return rmse_dict, pcc_dict
@torch.no_grad()
def test_step(model,loader,device):
    model.eval()
    y_true, preds = [], []
    for data in tqdm(loader):
        batch_sample = {k: v.to(device) for k, v in data.items()}   
        out_dict = model(batch_sample)
        y_true.append(batch_sample['label'])
        preds.append(out_dict['pred'].float().cpu())
    y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(preds, dim=0).cpu().detach().numpy()
    test_rmse = nn.MSELoss()(y_true, y_pred)
    rmse_dict = {}
    pcc_dict = {}
    r2_dict = {}
    MAE_dict = {}
    for i in range(7):
        column_true = y_true[:, i]
        column_pred = y_pred[:, i]
        rmse = mean_squared_error(column_true, column_pred, squared=False)
        rmse_dict[i] = rmse
        pcc = pearsonr(column_true, column_pred)[0]
        pcc_dict[i] = pcc
        r_2 = r2_score(column_true, column_pred)
        r2_dict[i] = r_2
        MAE = mean_absolute_error(column_true, column_pred)
        MAE_dict[i] = MAE
        print(f'Test accuracy for dose {i}: RMSE: {rmse:.4f}, PCC: {pcc:.4f}, R2: {r_2:.4f}, MAE: {MAE:.4f}')
    return rmse_dict, pcc_dict, r2_dict, MAE_dict, test_rmse

def train_multi_view_model(args, train_set, val_set, test_set):
    save_dir = 'best_model_' + args.train_type
    lr = args.lr
    batch_size = args.batch_size
    drug_embed_dim = args.drug_embed_dim
    dropout_rate = args.dropout_rate
    drug_layer_num = args.drug_layer_num   
    readout = args.readout ## mean, max
    JK = args.JK ## 'True', 'False', string value
    ## Config for cells
    cell_embed_dim = args.cell_embed_dim
    cell_layer_num = args.cell_layer_num
    ## Config for genes
    view_dim = args.view_dim   
    n_epochs = args.epochs 
    use_regulizer = args.use_regulizer
    use_drug_path_way = args.use_drug_path_way
    model_config = {'drug_embed_dim': drug_embed_dim,
                    'cell_embed_dim': cell_embed_dim, 
                    'hidden_dim': cell_embed_dim, 
                    'drug_layer_num': drug_layer_num, ## This is for drug
                    'cell_layer_num': cell_layer_num, ## This is for cell
                    'dropout_rate' : dropout_rate,
                    'readout': readout,
                    'JK': JK,
                    'view_dim': view_dim,
                    'use_regulizer': use_regulizer,
                    'use_drug_path_way': use_drug_path_way
                    }  
    path = f'./TB_5_fold/{save_dir}'+'.pth' 
    model = DRP_multi_view(model_config)
    # model = torch.compile(model)
    optimizer = opt.AdamW(model.parameters(), lr=lr, weight_decay= 0.01)
        # scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,num_warmup_steps=50, num_training_steps=n_epochs, lr_end = 1e-4, power=1)
    # elif optimizer_name == 'SGD': 
        # optimizer = opt.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)
    # cos_lr = lambda x : ((1+math.cos(math.pi* x /100) )/2)*(1-args.lrf) + args.lrf
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cos_lr)
    current_time = datetime.datetime.now().time()
    print('Begin Training')
    print(f'Embed_dim_drug : {drug_embed_dim}'+ '\n' +f'Hidden_dim_cell : {cell_embed_dim} \n' +  f'drug_layer_num : {drug_layer_num} \n'+ 
            f'read_out_function : {readout}\n'  +f'batch_size : {batch_size}\n' + f'view_dim : {view_dim}\n' + 
            f'lr : {lr}\n' + f'use_regulizer : {use_regulizer}\n' + f'use_drug_path_way : {use_drug_path_way}')
    tb = SummaryWriter(comment=current_time, log_dir=f'./TB_5_fold/{save_dir}')
    train_loader = DataLoader(train_set, batch_size= batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size= batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size= batch_size, shuffle=True, collate_fn=collate_fn)
    epoch_len = len(str(n_epochs))
    results = 'model_results'
    if os.path.exists(f"./{results}/{args.train_type}") is False:
        os.makedirs(f"./{results}/{args.train_type}")
    early_stop_count = 0 
    best_epoch = 0 
    best_val_rmse = 100
    if args.scheduler_type == 'OP':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience= 7 , verbose=True, min_lr= 0.05 * args.lr, factor= 0.1)
    elif args.scheduler_type == 'ML':
        scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1)
    for epoch in range(n_epochs):
        if early_stop_count < args.early_stop_count :
            train_rmse, train_pcc = train_step(model, train_loader, optimizer, tb, epoch, device, args)
            if args.scheduler_type == 'ML':
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            for i in range(7):
                print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '  + 
                            f'train_rmse for conc {i}: {train_rmse[i]:.5f} ' +
                            f'train_pcc for conc {i}: {train_pcc[i]:.5f} ' +  f'lr : {current_lr}')
                print(print_msg)
            # print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f}')
            if epoch % args.check_step == 0:
                val_rmse,val_pcc, val_r_2, val_mae, val_threshold_rmse = test_step(model, val_loader, device, args)
                if args.scheduler_type == 'OP':
                    scheduler.step(val_threshold_rmse)
                for i in range(7):
                    tb.add_scalar(f'Accuracy/val/pcc_conc_{i}', val_pcc[i], epoch)
                    tb.add_scalar(f"Accuracy/val/rmse_conc_{i}", val_rmse[i], epoch)
                    tb.add_scalar(f"Accuracy/val/mae_conc_{i}", val_mae[i], epoch)
                    tb.add_scalar(f"Accuracy/val/r_2_conc_{i}", val_r_2[i], epoch)
                    tb.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
                    print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '  + 
                                f'val_rmse for conc {i}: {val_rmse[i]:.5f} ' +
                                f'val_r_2 for conc {i}: {val_r_2[i]:.5f} ' +
                                f'val_mae for conc {i}: {val_mae[i]:.5f} ' +
                                f'val_pcc for conc {i}: {val_pcc[i]:.5f} ' +  f'lr : {current_lr}')
                    print(print_msg)
                if val_threshold_rmse < best_val_rmse:
                    early_stop_count = 0
                    best_val_rmse = val_threshold_rmse
                    best_epoch = epoch
                    test_rmse, test_pcc, test_r_2, test_mae, test_threshold_rmse = test_step(model,test_loader, device, args)
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict' : optimizer.state_dict(),
                            }, path)
                else: 
                    early_stop_count += 1 
                    print(f'Early stopping encounter : {early_stop_count}  times')
                if early_stop_count >= args.early_stop_count:
                    print('Early stopping!')
                    break
                print(f'Best epoch: {best_epoch:03d}')
                for i in range(7):
                    print(f'Best PCC for conc {i}: {test_pcc[i]:.4f},'
                        f'Best RMSE for conc {i}: {test_rmse[i]:.4f}, '
                        f'Best R_2 for conc {i}: {test_r_2[i]:.4f}, '
                        f'Best MAE for conc {i}: {test_mae[i]:.4f}')

    print("__________________________________________________________")
    hparams = {
        'use_regulizer': use_regulizer,
        'use_drug_path_way': use_drug_path_way,
        'best_epoch': best_epoch,
        'regular_weight': args.regular_weight,
        'regular_weight_drug_path_way': args.regular_weight_drug_path_way
    }

    metrics = {
        'val_threshold_rmse': best_val_rmse,
        'test_threshold_rmse': test_threshold_rmse
    }

    # Add metrics for each concentration
    for i in range(7):
        metrics[f'test_pcc_conc_{i}'] = test_pcc[i]
        metrics[f'test_rmse_conc_{i}'] = test_rmse[i]
        metrics[f'test_r2_conc_{i}'] = test_r_2[i]
        metrics[f'test_mae_conc_{i}'] = test_mae[i]
    tb.add_hparams(hparams, metrics)
    tb.close()

if __name__ == '__main__':
    args = arg_parse()
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu") 
    mut_tensor, chr_tensor, cna_tensor, GE_tensor, CL_type_tensor, label_tensor, result_df, drug2index, CL2index = preprocess_cell_data()
    drug_atom_feat, drug_bond_feat = process_drug_feat(drug2index)
    train_idx, val_idx, test_idx = train_val_test_split(result_df)
    drug_train_idx, CL_train_idx = process_CL_drug(train_idx)
    drug_val_idx, CL_val_idx = process_CL_drug(val_idx)
    drug_test_idx, CL_test_idx = process_CL_drug(test_idx)
    DRP_trainset = DRP_dataset(mut_tensor, chr_tensor, cna_tensor, GE_tensor, CL_type_tensor, drug_atom_feat, drug_bond_feat, result_tensor=label_tensor, drug_idx = drug_train_idx, CL_idx = CL_train_idx)
    DRP_valset = DRP_dataset(mut_tensor, chr_tensor, cna_tensor, GE_tensor, CL_type_tensor, drug_atom_feat, drug_bond_feat, result_tensor=label_tensor, drug_idx = drug_val_idx, CL_idx = CL_val_idx)
    DRP_testset = DRP_dataset(mut_tensor, chr_tensor, cna_tensor, GE_tensor, CL_type_tensor, drug_atom_feat, drug_bond_feat, result_tensor=label_tensor, drug_idx = drug_test_idx, CL_idx = CL_test_idx)
    train_multi_view_model(args, train_set = DRP_trainset, val_set = DRP_valset, test_set = DRP_testset)
    