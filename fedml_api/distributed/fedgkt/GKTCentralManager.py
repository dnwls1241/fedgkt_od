import logging
import os
from capstone.fedml_api.distributed.fedgkt.utils import save_dict_to_json
from fedml_api.distributed.fedgkt.GKTGlobalTrainer import GKTGlobalTrainer
from fedml_api.distributed.fedgkt.GKTLocalTrainer import GKTLocalTrainer
from fedml_api.data_preprocessing.coco.data_loader import init_distirbuted_data


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
        while self.round_idx < self.round_num:
            global_trainer = GKTGlobalTrainer(self.round_idx, self.args)
            for i in range(self.client_number):
                local_trainer = GKTLocalTrainer(i, self.args)
                global_trainer.add_local_trained_result(local_trainer.train())
                global_trainer.train_and_eval(self.round_idx)
            
            self.round_idx += 1
        

    def save_client_result(self, round_idx, i, extracted_feature_dict, logits_dict, labels_dict, extracted_feature_dict_test, labels_dict_test):
        
        

