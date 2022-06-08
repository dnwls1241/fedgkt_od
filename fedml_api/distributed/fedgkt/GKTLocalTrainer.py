import logging

import torch
from torch import nn, optim

from fedml_api.distributed.fedgkt import utils
from fedml_api.model.gkt.GKTRetinanet import gktclientmodel


class GKTLocalTrainer(object):
    def __init__(self, client_id, local_training_data, local_test_data, device,
                client_model, args):
        self.args = args
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.client_id = client_id
        self.device = device

        if client_model is None:
            client_model = gktclientmodel(pretrained=args.pretrained, path=args.path, \
                num_classes=args.num_classes, pretrained_backbone=args.pretrained_backbone)
        self.client_model = client_model
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
        
        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_KL = utils.KL_Loss(self.args.temperature)
        self.weight_path = args.weight_path
        self.server_logits_dict = dict()
    

    def update_large_model_logits(self, logits):
        self.server_logits_dict = logits

    def train(self):
        self.client_model.train()
        
        # key: batch_index; value: extracted_feature_map
        extracted_feature_dict = dict()

        # key: batch_index; value: logits
        logits_dict = dict()

        # key: batch_index; value: label
        labels_dict = dict()

         # for test - key: batch_index; value: extracted_feature_map
        extracted_feature_dict_test = dict()
        labels_dict_test = dict()

        if self.args.whether_training_on_client == 1:
            # train and update
            epoch_loss = []
            for epoch in range(self.args.epochs_client):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.local_training_data):
                    # images, labels = images.to(self.device), labels.to(self.device)
                    # logging.info("shape = " + str(images.shape))
                    if len(self.server_logits_dict) != 0:
                        large_model_logits = torch.from_numpy(self.server_logits_dict[batch_idx]).to(
                            self.device)
                        losses, logits, features = self.client_model(images, labels, large_model_logits)
                    else:
                        losses, logits, features = self.client_model(images, labels)

                    self.optimizer.zero_grad()
                    losses = sum(loss for loss in losses.values()) 
                    losses.backward()
                    self.optimizer.step()
                    logging.info('client {} - Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.client_id, epoch, batch_idx * len(images), len(self.local_training_data.dataset),
                                                  100. * batch_idx / len(self.local_training_data.dataset), losses.item()))
                    batch_loss.append(losses.item())
                    del images, labels
                torch.cuda.empty_cache()
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        self.save()

        self.client_model.eval()

        for batch_idx, (images, labels) in enumerate(self.local_training_data):
            # images, labels = images.to(self.device), labels.to(self.device)

            # logging.info("shape = " + str(images.shape))
            detections, log_probs, extracted_features = self.client_model(images)

            # logging.info("shape = " + str(extracted_features.shape))
            # logging.info("element size = " + str(extracted_features.element_size()))
            # logging.info("nelement = " + str(extracted_features.nelement()))
            # logging.info("GPU memory1 = " + str(extracted_features.nelement() * extracted_features.element_size()))
            extracted_feature_dict[batch_idx] = extracted_features.cpu().detach().numpy()
            log_probs = log_probs.cpu().detach().numpy()
            logits_dict[batch_idx] = log_probs
            labels_dict[batch_idx] = labels.cpu().detach().numpy()

        for batch_idx, (images, labels) in enumerate(self.local_test_data):
            test_images, test_labels = images.to(self.device), labels.to(self.device)
            _, extracted_features_test = self.client_model(test_images)
            extracted_feature_dict_test[batch_idx] = extracted_features_test.cpu().detach().numpy()
            labels_dict_test[batch_idx] = test_labels.cpu().detach().numpy()

        return extracted_feature_dict, logits_dict, labels_dict, extracted_feature_dict_test, labels_dict_test
    
    def save(self):
        torch.save(self.client_model, self.weight_path)
        print("save as", self.weight_path) 