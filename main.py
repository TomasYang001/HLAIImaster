import numpy as np

comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
from models import HLAIImaster
from time import time
from utils import set_seed, mkdir
from configs import get_cfg_defaults
from dataloader import Dataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="HLAIImaster for HLA-II epitope prediction")
parser.add_argument('--cfg', default="configs/HLAIImaster_DA.yaml", help="path to config file", type=str)
parser.add_argument('--data', default='immunogenicity', type=str, metavar='TASK',
                    help='dataset')
# parser.add_argument('--split', default='cluster', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster'])
args = parser.parse_args()

# def prepare_pMHC_pairs(HLA, Antigen,label):
#     _dataset = [[]]
#     for index in range(len(antigen)):
#         _dataset[index].append(antigen)
#         _dataset[index].append(HLA)
#         _dataset[index].append(label)
#         if index < len(antigen) - 1:
#             _dataset.append([])
#
#     return _dataset

def prepare_pMHC_pairs(HLA, Antigen, label):
    dataset = [[]]
    for ind in range(len(HLA)):
        hla = HLA[ind]
        dataset[ind].append(np.array(hla, dtype=np.float32))
        antigen = Antigen[ind]
        dataset[ind].append(np.array(antigen, dtype=np.float32))
        Y = label[ind]
        dataset[ind].append(np.array(Y, dtype=np.float32))
        if ind < len(HLA) - 1:
            dataset.append([])
    return dataset

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'

    if not cfg.DA.TASK:
        train_path = os.path.join(dataFolder, 'train')
        val_path = os.path.join(dataFolder, "val")
        test_path = os.path.join(dataFolder, "test")
        # df_train = pd.read_csv(train_path)
        # df_val = pd.read_csv(val_path)
        # df_test = pd.read_csv(test_path)

        train_epitope_antigen = np.load(train_path + '/antigen.npy')
        train_epitope_HLA = np.load(train_path + '/HLA.npy')
        train_epitope_label = np.load(train_path + '/label.npy')

        val_epitope_antigen = np.load(val_path + '/antigen.npy')
        val_epitope_HLA = np.load(val_path + '/HLA.npy')
        val_epitope_label = np.load(val_path + '/label.npy')

        test_epitope_antigen = np.load(test_path + '/antigen.npy')
        test_epitope_HLA = np.load(test_path + '/HLA.npy')
        test_epitope_label = np.load(test_path + '/label.npy')

        train_dataset = train_epitope_antigen, train_epitope_HLA, train_epitope_label
        val_dataset = val_epitope_antigen, val_epitope_HLA, val_epitope_label
        test_dataset = test_epitope_antigen, test_epitope_HLA, test_epitope_label
        # train_dataset = DTIDataset(df_train.index.values, df_train)
        # val_dataset = DTIDataset(df_val.index.values, df_val)
        # test_dataset = DTIDataset(df_test.index.values, df_test)
    else:

        train_antigenic_antigen = np.load(dataFolder + '/train_antigen.npy')
        train_antigenic_HLA = np.load(dataFolder + '/train_HLA.npy')
        train_antigenic_label =np.load(dataFolder + '/train_label.npy')
        train_antigenic_antigen = torch.FloatTensor(train_antigenic_antigen)
        train_antigenic_HLA = torch.FloatTensor(train_antigenic_HLA)
        print('train_antigenic_antigen:', train_antigenic_antigen.shape)

        immunogenic_antigen = np.load(dataFolder + '/IM_antigen.npy')
        immunogenic_HLA = np.load(dataFolder + '/IM_HLA.npy')
        immunogenic_label =np.load(dataFolder + '/IM_label.npy')
        immunogenic_antigen = torch.FloatTensor(immunogenic_antigen)
        immunogenic_HLA = torch.FloatTensor(immunogenic_HLA)
        print('immunogenic_antigen:', immunogenic_antigen.shape)
        
        train_immunogenic_antigen = immunogenic_antigen[:17536] # 4:1 for train and test ratio, 21920 in total
        train_immunogenic_HLA = immunogenic_HLA[:17536]
        train_immunogenic_label = immunogenic_label[:17536]

        test_immunogenic_antigen = immunogenic_antigen[17536:]
        test_immunogenic_HLA = immunogenic_HLA[17536:]
        test_immunogenic_label = immunogenic_label[17536:]

        train_dataset = prepare_pMHC_pairs(train_antigenic_HLA, train_antigenic_antigen, train_antigenic_label)
        train_CDAN_dataset = prepare_pMHC_pairs(train_immunogenic_HLA, train_immunogenic_antigen, train_immunogenic_label)
        test_CDAN_dataset = prepare_pMHC_pairs(test_immunogenic_HLA, test_immunogenic_antigen, test_immunogenic_label)

    if cfg.COMET.USE and comet_support:
        experiment = Experiment(
            project_name=cfg.COMET.PROJECT_NAME,
            workspace=cfg.COMET.WORKSPACE,
            auto_output_logging="simple",
            log_graph=True,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
            auto_metric_logging=False
        )

        
        hyper_params = {
            "LR": cfg.SOLVER.LR,
            "Output_dir": cfg.RESULT.OUTPUT_DIR,
            "DA_use": cfg.DA.USE,
            "DA_task": cfg.DA.TASK,
        }
        if cfg.DA.USE:
            da_hyper_params = {
                "DA_init_epoch": cfg.DA.INIT_EPOCH,
                "Use_DA_entropy": cfg.DA.USE_ENTROPY,
                "Random_layer": cfg.DA.RANDOM_LAYER,
                "Original_random": cfg.DA.ORIGINAL_RANDOM,
                "DA_optim_lr": cfg.SOLVER.DA_LR
            }
            hyper_params.update(da_hyper_params)
        experiment.log_parameters(hyper_params)
        if cfg.COMET.TAG is not None:
            experiment.add_tag(cfg.COMET.TAG)
        experiment.set_name(f"{args.data}_{suffix}")

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True}

    if not cfg.DA.USE:
        training_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        if not cfg.DA.TASK:
            val_generator = DataLoader(val_dataset, **params)
            test_generator = DataLoader(test_dataset, **params)
        else:
            val_generator = DataLoader(test_CDAN_dataset, **params)
            test_generator = DataLoader(test_CDAN_dataset, **params)
    else:
        antigentic_generator = DataLoader(train_dataset, **params)
        immunogenic_generator = DataLoader(train_CDAN_dataset, **params)
        print("111:", len(immunogenic_generator))
        n_batches = max(len(antigentic_generator), len(immunogenic_generator))
        print("n_batches:", n_batches)
        multi_generator = MultiDataLoader(dataloaders=[antigentic_generator, immunogenic_generator], n_batches=n_batches)
        params['shuffle'] = False
        params['drop_last'] = False
        val_generator = DataLoader(test_CDAN_dataset, **params)
        test_generator = DataLoader(test_CDAN_dataset, **params)

    model = HLAIImaster(**cfg).to(device)

    if cfg.DA.USE:
        if cfg["DA"]["RANDOM_LAYER"]:
            domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
        else:
            domain_dmm = Discriminator(input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"],
                                       n_class=cfg["DECODER"]["BINARY"]).to(device)
        # params = list(model.parameters()) + list(domain_dmm.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    if not cfg.DA.USE:
        trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, opt_da=None,
                          discriminator=None,
                          experiment=experiment, **cfg)
    else:
        trainer = Trainer(model, opt, device, multi_generator, val_generator, test_generator, opt_da=opt_da,
                          discriminator=domain_dmm,
                          experiment=experiment, **cfg)
    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")
    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")

