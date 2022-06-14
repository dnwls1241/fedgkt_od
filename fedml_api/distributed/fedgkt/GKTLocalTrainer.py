import logging

import torch
from torch import nn, optim
import os

import wandb
from fedml_api.distributed.fedgkt import utils
from fedml_api.model.gkt.GKTRetinanet import gktclientmodel
from .utils import load_client_result, save_client_result, load_server_logit, memory_usage

class GKTLocalTrainer(object):
    def __init__(self, client_id, round_idx, local_training_data, local_test_data, device,
                client_model=None, args=None):
        self.args = args
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.client_id = client_id
        self.device = device
        self.round_idx = round_idx
        self.resume_round = args.resume_round
        self.project_dir = args.project_dir
        self.client_dir = self.project_dir+"/client"
        self.server_dir = self.project_dir+"/server"
        self.train_large_model_logits = False
        self.weight_dir = args.weight_dir
        self.eval = args.clientmodel_eval
        if args.weight_dir is None:
            self.weight_dir = '.'
        self.weight_path = "/client/round{}_client{}.pth".format(self.round_idx, self.client_id)
        self.weight_path = self.weight_dir + self.weight_path
        
        if client_model is None:
            self.client_model = gktclientmodel(pretrained=args.pretrained, path=self.weight_path, \
                num_classes=args.num_classes, backbone_name=args.backbone_name)
        else:
            self.client_model = client_model
        
        if self.round_idx!=0:
            self.load_weight(self.round_idx-1)
        # else:
        #     self.load_weight(3)

        self.client_model.to(self.device)

        self.model_params = self.master_params = self.client_model.parameters()

        optim_params = utils.bnwd_optim_params(self.client_model, self.model_params,
                                                            self.master_params) if args.no_bn_wd else self.master_params

        if self.args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(optim_params, lr=self.args.lr, momentum=0.9,
                                             nesterov=True,
                                             weight_decay=self.args.wd)
        elif self.args.optimizer == "Adam":
            self.optimizer = optim.Adam(optim_params, lr=self.args.lr, weight_decay=0.0001, amsgrad=True)
        
        if round_idx != 0:
            optim_path = self.weight_dir + "/opt/client{}_round{}.pth".format(self.client_id, self.round_idx-1)
            if os.path.exists(optim_path):
                self.load_optimizer(optim_path)
        

    def update_large_model_logits(self, round_idx):
        
        self.round_idx = round_idx
        self.train_large_model_logits = True

    def train(self):
        self.client_model.train()
        # # key: batch_index; value: extracted_feature_map
        # extracted_feature_dict = dict()

        # # key: batch_index; value: logits
        # logits_dict = dict()

        # # key: batch_index; value: label
        # labels_dict = dict()

        #  # for test - key: batch_index; value: extracted_feature_map
        # extracted_feature_dict_test = dict()
        # labels_dict_test = dict()
        passing = False
        if os.path.exists(self.weight_dir + "/client/round{}_client{}.pth".format(self.round_idx, self.client_id)):
            passing = True

        if self.args.whether_training_on_client == 1 and not passing:
            # train and update
            epoch_loss = []
            for epoch in range(self.args.epochs_client):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.local_training_data):
                    # images, labels = images.to(self.device), labels.to(self.device)
                    # logging.info("shape = " + str(images.shape))
                    
                    if self.train_large_model_logits:
                        server_logits = load_server_logit(self.round_idx-1, self.server_dir, self.client_id, batch_idx)
                        large_model_logits = {k: torch.from_numpy(v).float().to(self.device) for k, v in server_logits.items()}
                        losses, _, _ = self.client_model(images, labels, large_model_logits)
                    else:
                        losses, _, _ = self.client_model(images, labels)

                    self.optimizer.zero_grad()
                    losses = sum(loss for loss in losses.values())
                    losses.backward()
                    self.optimizer.step()
                    if batch_idx%5 == 0:
                        logging.info('round {} client {} - Update Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                            self.round_idx, self.client_id, epoch, batch_idx * len(images), len(self.local_training_data.dataset), losses.item()))
                    batch_loss.append(losses.item())
                    del images, labels

                torch.cuda.empty_cache()
                loss = sum(batch_loss) / len(batch_loss)
                epoch_loss.append(loss)
                wandb.log({"client{}_Train/Loss".format(self.client_id): loss})
        if not passing:
            self.save_weight()
            self.save_optimizer()
        else:
            self.load_weight(self.round_idx)
        self.client_model.eval()
        if self.args.client_make_logits==1:
            for batch_idx, (images, labels) in enumerate(self.local_training_data):
                # images, labels = images.to(self.device), labels.to(self.device)

                # logging.info("shape = " + (str(i.shape) + " ") for i in images)
                detections, log_probs, extracted_features = self.client_model(images)

                # logging.info("shape = " + (str(p.shape) + " ") for p in extracted_features)
                # logging.info("element size = " + (str(p.element_size()) + " ") for p in extracted_features)
                # logging.info("nelement = " + (str(p.nelement())+ " ") for p in extracted_features)
                # logging.info("GPU memory1 = " + (str(p.nelement() * p.element_size())+ " ") for p in extracted_features)
                # extracted_features = [p.cpu().detach().numpy() for p in extracted_features]
                logits_dict = {k: v.cpu().detach().numpy() for k, v in log_probs.items()}
                # labels_dict = [{k: v.cpu().detach().numpy() for k, v in t.items()} for t in labels]
                # train_result = {"features":extracted_features,
                #                 "logits": logits_dict, 
                #                 "labels": labels_dict}
                if batch_idx%5 == 0: logging.info('round{} client{} train_batch[{}/{}] '.format(self.round_idx, self.client_id, batch_idx, len(self.local_training_data)))
                save_client_result(self.round_idx, self.client_dir, self.client_id, batch_idx, logits_dict)
                
                del images, labels, extracted_features, logits_dict, log_probs
                torch.cuda.empty_cache()
            train_batch_num = batch_idx+1
        
        if self.eval==1:
            for batch_idx, (test_images, test_labels) in enumerate(self.local_test_data):
                _, _, extracted_features_test = self.client_model(test_images)
                extracted_features_test = [p.cpu().detach().numpy() for p in extracted_features_test]
                labels_dict_test = [{k: v.cpu().detach().numpy() for k, v in t.items()} for t in test_labels]
                test_result = {"features":extracted_features_test, 
                                "labels": labels_dict_test}
                
                if batch_idx%5 == 0: logging.info('test_batch{}/{} clear'.format(batch_idx, len(self.local_test_data)))
                # save_client_result(self.round_idx, self.test_dir, self.client_id, batch_idx, test_result)
                del test_images, test_labels, extracted_features_test, labels_dict_test, test_result
                torch.cuda.empty_cache()
        print("complete")

    def save_weight(self):
        self.weight_path = self.weight_dir + "/client/round{}_client{}.pth".format(self.round_idx, self.client_id)
        torch.save(self.client_model.state_dict(), self.weight_path)
        print("save model as", self.weight_path) 

    def load_weight(self, round_idx):
        self.weight_path = self.weight_dir + "/client/round{}_client{}.pth".format(round_idx, self.client_id)
        self.client_model.load_state_dict(torch.load(self.weight_path))
        print("load model", self.weight_path) 
    
    def save_optimizer(self):
        path = self.weight_dir + "/opt/client{}_round{}.pth".format(self.client_id, self.round_idx)
        torch.save(self.optimizer.state_dict(), path)
        print("save client optimizer as", path) 
    
    def load_optimizer(self, path):
        self.optimizer.load_state_dict(torch.load(path))
        print("load client optimizer ", path) 