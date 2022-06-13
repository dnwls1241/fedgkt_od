import logging
import os
import pickle
from capstone.fedml_api.distributed.fedgkt.utils import save_dict_to_json
from fedml_api.distributed.fedgkt.GKTGlobalTrainer import GKTGlobalTrainer
from fedml_api.distributed.fedgkt.GKTLocalTrainer import GKTLocalTrainer
from fedml_api.data_preprocessing.coco.data_loader import init_distirbuted_data
from .utils import load_client_result, load_server_logit, save_client_result, save_server_logit

class GKTCentralManager(object):
    def __init__(self, args):
        self.args = args
        self.round_num = args.comm_round
        self.round_idx = 0
        self.data_dir = args.data_dir
        self.client_number = args.client_number
        self.dataidx_map = init_distirbuted_data(args)
        self.project_dir = args.project_dir
        
    def run(self):
        root = self.project_dir
        for round_idx in range(self.round_num):
            
            global_trainer = GKTGlobalTrainer(self.args)
            
            for i in range(self.client_number):
                local_trainer = GKTLocalTrainer(i, self.args)
                if round_idx > 0:
                    logit = load_server_logit(round_idx, i)
                    local_trainer.update_large_model_logits(logit)
                client_result = local_trainer.train()
                save_client_result(round_idx, i, client_result)
            epochs = global_trainer.get_server_epoch_strategy2(round_idx)
            global_trainer.train_and_eval(round_idx, epochs)