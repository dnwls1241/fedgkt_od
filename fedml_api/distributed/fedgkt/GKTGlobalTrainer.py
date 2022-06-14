import logging
import wandb
import torch
import os
import shutil
from torch import nn, optim
from fedml_api.distributed.fedgkt import utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fedml_api.model.gkt.GKTRetinanet import gktservermodel
from fedml_api.data_preprocessing.coco.data_loader import get_dataloader_coco_v2
from .utils import load_client_result, save_server_logit
from pycocotools.cocoeval import COCOeval


class GKTGlobalTrainer(object):
    def __init__(self, args, round_idx, data_map, test_data_idxs, last_epoch, model=None):
        self.client_num = args.client_number
        self.device = args.device
        self.args = args
        self.round_idx = round_idx
        self.project_dir = args.project_dir
        self.client_dir = self.project_dir+"/client"
        self.server_dir = self.project_dir+"/server"
        self.last_epoch = last_epoch
        self.data_map = data_map
        self.test_data_idxs =  test_data_idxs
        self.eval = args.servermodel_eval
        self.weight_dir = args.weight_dir
        """
            when use data parallel, we should increase the batch size accordingly (single GPU = 64; 4 GPUs = 256)
            One epoch training time: single GPU (64) = 1:03; 4 x GPUs (256) = 38s; 4 x GPUs (64) = 1:00
            Note that if we keep the same batch size, the frequent GPU-CPU-GPU communication will lead to
            slower training than a single GPU.
        """
        # server model
        if args.weight_dir is None:
            self.weight_dir = '.'
        self.weight_path = "/server/round{}_epoch{}.pth".format(self.round_idx, self.last_epoch)
        self.weight_path = self.weight_dir + self.weight_path
        
        if model is not None:
            self.model_global = model
        else:
            self.model_global = gktservermodel(pretrained=args.pretrained, path=self.weight_path,
                                            num_classes=args.num_classes, backbone_name=args.backbone_name, server_chan=args.server_chan, cls_alpha=args.alpha, reg_alpha=args.alpha)
        if self.round_idx != 0:
            self.load_weight(self.round_idx-1, self.last_epoch)
        self.model_global.to(self.device)
        if args.multi_gpu_server and torch.cuda.device_count() > 1:
            device_ids = [i for i in range(torch.cuda.device_count())]
            self.model_global = nn.DataParallel(self.model_global, device_ids=device_ids)

        self.model_global.train()
       

        self.model_params = self.master_params = self.model_global.parameters()

        optim_params = utils.bnwd_optim_params(self.model_global, self.model_params,
                                                            self.master_params) if args.no_bn_wd else self.master_params

        if self.args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(optim_params, lr=self.args.lr, momentum=0.9,
                                             nesterov=True,
                                             weight_decay=self.args.wd)
        elif self.args.optimizer == "Adam":
            self.optimizer = optim.Adam(optim_params, lr=self.args.lr, weight_decay=0.0001, amsgrad=True)

        if self.round_idx != 0:
            optim_path = self.weight_dir + "/opt/server_round{}_epoch{}.pth".format(self.round_idx-1, self.last_epoch)
            if os.path.exists(optim_path):
                self.load_optimizer(optim_path)
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max')
        # self.criterion_CE = nn.CrossEntropyLoss()
        # self.criterion_KL = utils.KL_Loss(self.args.temperature)
        # self.best_acc = 0.0

        # key: client_index; value: extracted_feature_dict
        # for test

    # def add_local_trained_result(self, index, extracted_feature_dict, logits_dict, labels_dict,
    #                              extracted_feature_dict_test, labels_dict_test):
    #     logging.info("add_model. index = %d" % index)
    #     self.client_extracted_feauture_dict[index] = extracted_feature_dict
    #     self.client_logits_dict[index] = logits_dict
    #     self.client_labels_dict[index] = labels_dict
    #     self.client_extracted_feauture_dict_test[index] = extracted_feature_dict_test
    #     self.client_labels_dict_test[index] = labels_dict_test

    #     self.flag_client_model_uploaded_dict[index] = True

    def set_round_idx(self, round_idx):
        self.round_idx = round_idx
    
    def get_round_idx(self, round_idx):
        return self.round_idx

    def get_clients_batch_num(self, batch_num):
        self.clients_batch_num = batch_num

    def get_server_epoch_strategy_test(self):
        return 1, True
     # ResNet56
    def get_server_epoch_strategy_reset56(self, round_idx):
        whether_distill_back = True
        # set the training strategy
        if round_idx < 20:
            epochs = 20
        elif 20 <= round_idx < 30:
            epochs = 15
        elif 30 <= round_idx < 40:
            epochs = 10
        elif 40 <= round_idx < 50:
            epochs = 5
        elif 50 <= round_idx < 100:
            epochs = 5
        elif 100 <= round_idx < 150:
            epochs = 3
        elif 150 <= round_idx <= 200:
            epochs = 2
            whether_distill_back = False
        else:
            epochs = 1
            whether_distill_back = False
        return epochs, whether_distill_back
    
    def get_server_epoch_strategy_reset56_2(self, round_idx):
        whether_distill_back = True
        # set the training strategy
        epochs = self.args.epochs_server
        return epochs, whether_distill_back
    
    # not increase after 40 epochs
    def get_server_epoch_strategy2(self, round_idx):
        whether_distill_back = True
        # set the training strategy
        if round_idx < 20:
            epochs = 5
        elif 20 <= round_idx < 30:
            epochs = 5
        elif 30 <= round_idx < 40:
            epochs = 5
        elif 40 <= round_idx < 50:
            epochs = 5
        elif 50 <= round_idx < 100:
            epochs = 4
        elif 100 <= round_idx < 150:
            epochs = 3
        elif 150 <= round_idx <= 200:
            epochs = 1
            whether_distill_back = False
        else:
            epochs = 1
            whether_distill_back = False
        return epochs, whether_distill_back

    def train_and_eval(self, round_idx, epochs):
        self.round_idx = round_idx
        for epoch in range(epochs):
            logging.info("train_and_eval. round_idx = %d, epoch = %d" % (round_idx, epoch))
            if self.args.whether_training_on_server == 1: 
                train_metrics = self.train_large_model_on_the_server()
            self.save_weight(epoch)
            self.save_optimizer(epoch)
            
            wandb.log({"Train/Loss": train_metrics['train_loss'], "round":epoch+1, "epoch": round_idx + 1})
            if epoch == epochs - 1:
                if self.args.server_make_logits==1:
                    self.make_logits_large_model()
                print("server logit made")
                shutil.rmtree(self.client_dir)
                os.mkdir(self.client_dir)
                if self.eval == 1:
                    test_metrics = self.eval_large_model_on_the_server()
                print("server model eval")
                
                return epoch

    def train_large_model_on_the_server(self):
        # clear the server side logits
        self.model_global.train()

        loss_avg = utils.RunningAverage()
        
        for client_id in range(self.client_num):
            train_dl, _ = get_dataloader_coco_v2(self.args, self.data_map[str(client_id)], self.test_data_idxs)

            for batch_index, (images, labels) in enumerate(train_dl):
                
                logits = {}
                if self.args.whether_distill_on_the_server == 1:
                    logits = load_client_result(self.round_idx, self.client_dir, client_id, batch_index)
                    logits = {k: torch.from_numpy(v).float().to(self.device) for k, v in logits.items()}
                    losses, _ = self.model_global(images, labels, logits)
                else:
                    losses, _ = self.model_global(images, labels)

                self.optimizer.zero_grad()
                losses = sum(loss for loss in losses.values())
                losses.backward()
                self.optimizer.step()

                if batch_index%5 == 0:
                        logging.info('Server training: round {} client {} [{}/{}]\tLoss: {:.6f}'.format(
                            self.round_idx, client_id, batch_index * len(images), len(train_dl.dataset), losses.item()))
                loss_avg.update(losses.item())
                del images, labels, logits
                torch.cuda.empty_cache()
           
            del train_dl
            torch.cuda.empty_cache()
        train_metrics = {'train_loss': loss_avg.value()}

        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
        logging.info("- Train metrics: " + metrics_string)

        return train_metrics
    
    def eval_large_model_on_the_server(self):

        # set model to evaluation mode
        self.model_global.eval()
        # loss_avg = utils.RunningAverage()
        # accTop1_avg = utils.RunningAverage()
        # accTop5_avg = utils.RunningAverage()
        with torch.no_grad():
            for client_id in range(self.client_num):
                _, test_dl = get_dataloader_coco_v2(self.args, self.data_map[str(client_id)], self.test_data_idxs)
                for batch_index, (images, labels) in enumerate(test_dl):
                    logits = load_client_result(self.round_idx, self.client_dir, client_id, batch_index)
                    logits = {k: torch.from_numpy(v).float().to(self.device) for k, v in logits.items()}
                    detections, _, _ = self.model_global(images, labels)
                    server_result = detections.cpu().detach().numpy()
                    del images, labels, server_result
                    torch.cuda.empty_cache()
                    # Update average loss and accuracy
                    # metrics = utils.accuracy(detections, labels, topk=(1, 5))
                    # # only one element tensors can be converted to Python scalars
                    # accTop1_avg.update(metrics[0].item())
                    # accTop5_avg.update(metrics[1].item())
            del test_dl
            torch.cuda.empty_cache()
            
        # compute mean of all metrics in summary
        # test_metrics = {'test_accTop1': accTop1_avg.value(),
        #                 'test_accTop5': accTop5_avg.value()}

        # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
        # logging.info("- Test  metrics: " + metrics_string)
        test_metrics = {}
        return test_metrics

    def make_logits_large_model(self):
        self.model_global.eval()
        # loss_avg = utils.RunningAverage()
        # accTop1_avg = utils.RunningAverage()
        # accTop5_avg = utils.RunningAverage()
        with torch.no_grad():
            for client_id in range(self.client_num):
                train_dl, _ = get_dataloader_coco_v2(self.args, self.data_map[str(client_id)], self.test_data_idxs)
                for batch_index, (images, labels) in enumerate(train_dl):
                    losses, output_logits = self.model_global(images, labels, eval_loss=True)
                    output_logits = {k: v.cpu().detach().numpy() for k, v in output_logits.items()}
                    save_server_logit(self.round_idx, self.server_dir, client_id, batch_index, output_logits)
                    if batch_index%5 == 0:
                        logging.info('Server infer logit: round {} client {} [{}/{}]\tLoss: {:.6f}'.format(
                            self.round_idx, client_id, batch_index * len(images), len(train_dl.dataset), sum(loss for loss in losses.values())))

                    del labels, images, output_logits
                    torch.cuda.empty_cache()
                del train_dl
                torch.cuda.empty_cache()
    
    def make_logits_large_model_one_client(self, client_id):
        self.model_global.eval()
        # loss_avg = utils.RunningAverage()
        # accTop1_avg = utils.RunningAverage()
        # accTop5_avg = utils.RunningAverage()
        with torch.no_grad():
            train_dl, _ = get_dataloader_coco_v2(self.args, self.data_map[str(client_id)], self.test_data_idxs)
            for batch_index, (images, labels) in enumerate(train_dl):
                losses, output_logits = self.model_global(images, labels, eval_loss=True)
                output_logits = {k: v.cpu().detach().numpy() for k, v in output_logits.items()}
                save_server_logit(self.round_idx, self.server_dir, client_id, batch_index, output_logits)
                if batch_index%5 == 0:
                    logging.info('Server infer logit: round {} client {} [{}/{}]\tLoss: {:.6f}'.format(
                        self.round_idx, client_id, batch_index * len(images), len(train_dl.dataset), sum(loss for loss in losses.values())))

                del labels, images, output_logits
                torch.cuda.empty_cache()
            del train_dl
            torch.cuda.empty_cache()

    def save_weight(self, epoch):
        self.weight_path = self.weight_dir + "/server/round{}_epoch{}.pth".format(self.round_idx, epoch)
        torch.save(self.model_global.state_dict(), self.weight_path)
        print("save as", self.weight_path) 
    
    def load_weight(self, round_idx, epoch):
        self.weight_path = self.weight_dir + "/server/round{}_epoch{}.pth".format(round_idx, epoch)
        self.model_global.load_state_dict(torch.load(self.weight_path))
    
    def save_optimizer(self, epoch):
        path = self.weight_dir + "/opt/server_round{}_epoch{}.pth".format(self.round_idx, epoch)
        torch.save(self.optimizer.state_dict(), path)
        print("save server optimizer as", path) 
    
    def load_optimizer(self, path):
        self.optimizer.load_state_dict(torch.load(path))
        print("load server optimizer ", path) 