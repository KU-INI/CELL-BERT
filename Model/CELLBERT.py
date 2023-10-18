from .BertModel import BertModel
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from .activation import ACT2FN
from .ForPretrainedModel import BertPreTrainedModel
from .loss import FocalLoss

import copy
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.cluster import BisectingKMeans
from sklearn.mixture import BayesianGaussianMixture
import numpy as np

def get_extended_attention_mask(attention_mask, input_shape, device = None, dtype = torch.float32):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
        
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask

class CELLBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config


        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        #cluster and additional modules
        self.cluster = None
        self.encoder_ = None
        self.pooler_ = None
        self.classifier_ = None
        self.cluster_number = 2
        
        #for cluster
        self.save_hidden = torch.Tensor([])
        self.cluster_labelweight = None
        self.label = torch.Tensor([])
        self.softmax = nn.Softmax(dim = 1)
        self.post_init()
    
    """
    Create additional model and cluster. You can specify a cluster through "self.config.cluster_name".
            "KMeans",
            "GaussianMixture..diag.random-from-data",
            "GaussianMixture..diag.k-means++",
            "GaussianMixture..diag.k-means",
            "GaussianMixture..tied.k-means++",
            "GaussianMixture..tied.random",
            "GaussianMixture..tied.random-from-data",
            "GaussianMixture..full.random",
            "Birch",
            "BisectingKMeans..lloyd.largest-cluster.k-means++",
            "BayesianGaussianMixture..diag.random-from-data",
            "BayesianGaussianMixture..full.random",
            "BayesianGaussianMixture..full.random-from-data"
            "BayesianGaussianMixture..full.k-means++",
            "BayesianGaussianMixture..tied.k-means++",
            "BayesianGaussianMixture..tied.random",
            "BayesianGaussianMixture..tied.random-from-data",
            and so on..
    """
    def make_sub_model(self, N_CLUSTER = 2, sub_mode = None):
        self.cluster_number = N_CLUSTER
        
        self.encoder_ = nn.ModuleList([copy.deepcopy(sub_mode.bert.encoder) for _ in range(self.cluster_number)])
        self.pooler_ = nn.ModuleList([copy.deepcopy(sub_mode.bert.pooler) for _ in range(self.cluster_number)])
        self.classifier_ = nn.ModuleList([copy.deepcopy(sub_mode.classifier) for _ in range(self.cluster_number)])
        #label weight
        weight = []
        for i in range(self.config.num_labels):
            weight.append(0)
        self.cluster_labelweight = [copy.deepcopy(weight) for i in range(self.cluster_number)]
        
        
        if self.config.cluster_name == "KMeans":
            self.cluster = KMeans(n_clusters = N_CLUSTER, n_init = 10)
            #training cluster
            self.cluster.fit(self.save_hidden)
            predict = self.cluster.predict(self.save_hidden)
            
        elif "GaussianMixture" in self.config.cluster_name:
            covariance_type_ = self.config.cluster_name.split('.')[2]
            init_params_ = self.config.cluster_name.split('.')[3]
            if init_params_ == "random-from-data":
                init_params_ = "random_from_data"
            if "Bayesian" in self.config.cluster_name:
                self.cluster = BayesianGaussianMixture(n_components=N_CLUSTER, 
                                                      random_state=42, 
                                                      covariance_type = covariance_type_,
                                                      init_params = init_params_
                                                      )
            else:
                self.cluster = GaussianMixture(n_components=N_CLUSTER, 
                                              random_state=0, 
                                              covariance_type = covariance_type_,
                                              init_params = init_params_
                                              )
            #training cluster
            self.cluster.fit(self.save_hidden)
            predict = self.cluster.predict(self.save_hidden)
        
        elif self.config.cluster_name == "Birch":
            self.cluster = Birch(n_clusters=N_CLUSTER)
            #training cluster
            self.cluster.fit(self.save_hidden)
            predict = self.cluster.predict(self.save_hidden)
        
        elif "BisectingKMeans" in self.config.cluster_name:
            algorithm_ = self.config.cluster_name.split('.')[2]
            bisecting_strategy_ = self.config.cluster_name.split('.')[3]
            if bisecting_strategy_ == "largest-cluster":
                bisecting_strategy_ = "largest_cluster"
            init_ = self.config.cluster_name.split('.')[4]
            self.cluster = BisectingKMeans(n_clusters=2, 
                                           random_state=42,
                                           n_init = 10, 
                                           algorithm = algorithm_,
                                           bisecting_strategy = bisecting_strategy_,
                                           init = init_)
            #training cluster
            self.cluster.fit(self.save_hidden)
            predict = self.cluster.predict(self.save_hidden)
            
        ###############################
        #Create weight for Focal Loss#
        ##############################
        for i in range(len(self.cluster_labelweight)):
            total = len(predict[np.where(predict == i)])
            for j in self.label[np.where(predict == i)]:
                self.cluster_labelweight[i][int(j)] += 1
            for j in range(len(self.cluster_labelweight[0])):
                self.cluster_labelweight[i][j] = 1 - (self.cluster_labelweight[i][j] / total) + 0.001
                
        del self.save_hidden
        del self.label
        
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        labels = None,
        active_cluster = False, #You must run "make_sub_model" before you run the cluster
        save_hidden = False,
        mode = "train",
        weighted = [0.1, 0.85],
        device = None
    ):
        logits_ = [None for _ in range(self.cluster_number)]
        
        with torch.no_grad():
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                save_hidden = True
            )
            #embedding output of base model for additional model's input
            embedding_output = outputs[0]
            encoder_output = outputs[1]
            pooled_output = outputs[2]
            hidden_cls = outputs[3] #cls token on all layers for data clustering
             
            
            ######################################################
            #save [AVG_CLS] token for training cluster           #
            #"hidden_cls" is used for data clustering.           #
            #and "self.save_hidden" is used for training cluster.#
            ######################################################
            if save_hidden:
                average_cls = None
                for cls in hidden_cls:
                    if average_cls == None:
                        average_cls = cls
                    else:
                        average_cls += cls
                average_cls /= len(hidden_cls)
                del hidden_cls
                self.save_hidden = torch.cat([self.save_hidden, 
                                              average_cls.clone().detach().to('cpu')])
                
                #save label for weighted focal loss
                self.label = torch.cat([self.label, 
                                              labels.clone().detach().to('cpu')])
            
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
        input_shape = input_ids.size()
        extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape)
        if active_cluster and self.cluster != None:
            ###############################
            #[AVG_CLS] for data clustering#
            ###############################
            average_cls = None
            for cls in hidden_cls:
                if average_cls == None:
                    average_cls = cls
                else:
                    average_cls += cls
            average_cls /= len(hidden_cls)
            del hidden_cls
            try:
                cluster_result = self.cluster.predict(average_cls.clone().detach().to('cpu'))
            except:
                cluster_result = self.cluster.fit_predict(average_cls.clone().detach().to('cpu'))
                
                
            for k in range(self.cluster_number):
                if mode == "train":
                    encoder_outputs = self.encoder_[k](embedding_output[np.where(cluster_result == k)],
                                                       attention_mask=extended_attention_mask[np.where(cluster_result == k)])
                    pooled_output = self.pooler_[k](encoder_outputs[0])
                    pooled_output = self.dropout(pooled_output)
                    logits_[k] = self.classifier_[k](pooled_output)
                    
                elif mode == "validation":
                    with torch.no_grad():
                        encoder_outputs = self.encoder_[k](embedding_output,
                                                           attention_mask=extended_attention_mask)
                        pooled_output = self.pooler_[k](encoder_outputs[0])
                        pooled_output = self.dropout(pooled_output)
                        logits_[k] = self.classifier_[k](pooled_output)
                        
            ##############################################################################################################
            #weighted[0] is weight for base model, weighted[1] is in-cluster weight, weighted[2] is out-of-cluster weight#
            ##############################################################################################################
            if mode == "validation":
                for k in range(self.cluster_number):
                    # weighted[0] is weight for base model
                    logits[np.where(cluster_result == k)] = self.softmax(logits[np.where(cluster_result == k)]) * weighted[0]
                    for t in range(self.cluster_number):
                        if k == t:
                            # weighted[1] is in-cluster weight
                            logits[np.where(cluster_result == k)] += (self.softmax(logits_[t][np.where(cluster_result == k)]) * weighted[1])
                        else:
                            # weighted[2] == (1 - weighted[0] - weighted[1]) is out-of-cluster weight
                            logits[np.where(cluster_result == k)] += (self.softmax(logits_[t][np.where(cluster_result == k)]) * \
                            ((1 - weighted[0] - weighted[1]) / (self.cluster_number - 1))) 
                    
        loss = None
        weights = None
        if self.cluster_labelweight != None:
            weights = torch.FloatTensor(self.cluster_labelweight).to(device)

        if labels is not None:
            if weights != None:
                for i in range(len(weights)):
                    ####################
                    #Weighted FocalLoss#
                    ####################
                    loss_fct = FocalLoss(alpha = 0.25, gamma = 2, weight = weights[i])
                    if len(np.where(cluster_result == i)[0]) != 0:
                        if loss == None:
                            loss = loss_fct(logits_[i].view(-1, self.num_labels), labels[np.where(cluster_result == i)].view(-1))
                        else:
                            loss += loss_fct(logits_[i].view(-1, self.num_labels), labels[np.where(cluster_result == i)].view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
           
        output = (logits, outputs[0], outputs[1], outputs[2]) + outputs[3:]
        return ((loss,) + output) if loss is not None else output

 