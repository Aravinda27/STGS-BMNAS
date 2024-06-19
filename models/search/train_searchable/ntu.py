from cgi import test
import collections
from gc import collect
import torch
import models.auxiliary.scheduler as sc
import copy
from tqdm import tqdm
import os
from models.search.darts.utils import count_parameters, save, save_pickle
import numpy as np
from IPython import embed
from collections import Counter
import pickle
from sklearn import metrics
from models.search.darts.utils import AvgrageMeter

train_model_save = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/BM-NAS-master/train_model/temp/found"
auc_data_path = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/BM-NAS-master/train_model/temp/found/auc_data.txt"
train_loss_path = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/BM-NAS-master/train_model/temp/found/train_loss.txt"
dev_loss_path = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/BM-NAS-master/train_model/temp/found/dev_loss.txt"
test_loss_path = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/BM-NAS-master/train_model/temp/found/test_loss.txt"
operation_file_path = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/BM-NAS-master/train_model/temp/found"
entropy_file_path = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/BM-NAS-master/train_model/temp/found/gumbelsoftmax"

# train model with darts mixed operations
def train_ntu_track_acc(model, architect, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
                device=None, num_epochs=200, verbose=False, parallel=False, logger=None,
                plotter=None, args=None, status='search'):

    best_genotype = None
    best_acc = 0
    best_epoch = 0
    all_genotype = []
    all_acc = []
    all_epoch = []
    all_model = []
    best_test_genotype = None
    best_test_acc = 0
    best_test_epoch = 0
    train_loss = []
    test_loss = []
    val_loss = []
    final_layer_val = []
    # best_test_model_sd = copy.deepcopy(model.state_dict())
    
    num_input_nodes = args.num_input_nodes
    num_keep_edges = args.num_keep_edges
    node_steps = args.node_steps

    flag = 1
    op = ['Sum','ScaleDotAttn','LinearGLU','ConcatFC']
    all_op_count = dict()
    all_op_c1_count = dict()
    all_op_c2_count = dict()
    for i in op:
        all_op_count[i] = 0
        all_op_c1_count[i] = 0
        all_op_c2_count[i] = 0

    alpha_entropies = AvgrageMeter()
    gamma_entropies = AvgrageMeter()
    alpha_entropy_list = []
    gamma_entropy_list = []
    
    for epoch in range(num_epochs):
        pred_labels = []
        actual_labels = []
        pred_ind = []
        labels_data = []
        # Each epoch has a training and validation phase
        logger.info("Epoch: {}".format(epoch) )
        logger.info("EXP: {}".format(args.save) )

        phases = []
        if status == 'search':
            phases = ['train', 'dev']
        else:
            phases = ['train', 'test']
            # model.load_state_dict(best_test_model_sd)

        for phase in phases:

            if phase == 'train':
                if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                    scheduler.step()
                model.train()  # Set model to training mode
            elif phase == 'dev':
                model.train()
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0


            with tqdm(dataloaders[phase]) as t:

                # Iterate over data.
                for data in dataloaders[phase]:

                    # get the inputs
                    rgbs, spec, labels = data['rgb'], data['spec'], data['label']

                    # device
                    rgbs = rgbs.to(device)
                    spec = spec.to(device)
                    labels = labels.to(device)
                    #x = labels.item()
                    if phase == 'test':
                        labels_data.extend(labels.data)
                        actual_labels.extend(labels.tolist())
                    # continue

                    input_features = (rgbs, spec)
                    # updates darts cell
                    if status == 'search' and (phase == 'dev' or phase == 'test'):
                    # if status == 'search':
                    # if phase == 'test':
                        if architect is not None:
                            architect.step(input_features, labels, logger)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train' or (phase == 'dev' and status == 'eval')):
                        
                        # alpha_entropies.update(model.fusion_net.compute_arch_entropy().mean(),rgbs.size(0))

                        output = model(input_features)
                        
                        pred_val, preds = torch.max(output, 1)
                        if epoch == num_epochs-1 and phase == 'test':
                            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                            # print(len(output))
                            for i in range(0, len(output)):
                                #print(output[i])
                                val_final = output[i].tolist()
                                val_final.append(labels[i].tolist())
                                final_layer_val.append(val_final)
                            # print(len(pred_val))
                            # for i in range(0, len(pred_val)):
                            #     print(pred_val[i])
                            #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

                        if phase == 'test':
                            pred_ind.extend(preds.tolist())
                            pred_labels.extend(pred_val.tolist())
                

                        loss = criterion(output, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train' or (phase == 'dev' and status == 'eval'):
                            if isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                                scheduler.step()
                                scheduler.update_optimizer(optimizer)
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * rgbs.size(0)
                    #pred_labels = preds == labels.data
                    running_corrects += torch.sum(preds == labels.data)

                    batch_acc = torch.sum(preds == labels.data) * 1.0 / rgbs.size(0) 
                    postfix_str = 'batch_loss: {:.03f}, batch_acc: {:.03f}'.format(loss.item(), batch_acc)

                    t.set_postfix_str(postfix_str)
                    t.update()


            ######################## Entropy calculation #####################################
            
            # print("alpha entropy:", model.fusion_net.compute_arch_entropy().mean().item())
            # new_temp = alpha_entropies.avg.item()
            # print("alpha entropy average:",new_temp)
            # alpha_entropy_list.append(new_temp)
            # gamma_parameter = model.fusion_net.cell._step_nodes
            # for i in range(args.steps):
            #     gamma_entropies.update(gamma_parameter[i].compute_arch_entropy_gamma().mean(),1)
            # new_temp2 = gamma_entropies.avg.item()
            # print("gamma entropy average:",new_temp2)
            # gamma_entropy_list.append(new_temp2)

            ######################################################################################

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                train_loss.append(epoch_loss)
            if phase == 'test':
                test_loss.append(epoch_loss)
            if phase == 'dev':
                val_loss.append(epoch_loss)
            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            genotype = None


            if parallel:
                num_params = 0
                for reshape_layer in model.module.reshape_layers:
                    num_params += count_parameters(reshape_layer)

                num_params += count_parameters(model.module.fusion_net)
                logger.info("Fusion Model Params: {}".format(num_params) )

                genotype = model.module.genotype()
            else:
                num_params = 0
                for reshape_layer in model.reshape_layers:
                    num_params += count_parameters(reshape_layer)

                num_params += count_parameters(model.fusion_net)
                logger.info("Fusion Model Params: {}".format(num_params) )

                genotype = model.genotype()
            logger.info(str(genotype))
            if(phase == 'dev'):
                all_genotype.append(genotype)
                all_acc.append(epoch_acc.item())
                all_epoch.append(epoch)
                all_model.append(model)
            # deep copy the model
            if phase == 'dev' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                # best_model_sd = copy.deepcopy(model.state_dict())
                best_genotype = copy.deepcopy(genotype)
                best_epoch = epoch

                if parallel:
                    save(model.module, os.path.join(args.save, 'best', 'best_model.pt'))
                else:
                    save(model, os.path.join(args.save, 'best', 'best_model.pt'))

                best_genotype_path = os.path.join(args.save, 'best', 'best_genotype.pkl')
                save_pickle(best_genotype, best_genotype_path)
            
            # deep copy the model
            if phase == 'test' and epoch_acc >= best_test_acc:
                best_test_acc = epoch_acc
                # best_test_model_sd = copy.deepcopy(model.state_dict())
                best_test_genotype = copy.deepcopy(genotype)
                best_test_epoch = epoch
                print("######## test accuracy: ",best_test_acc," #############")
                if parallel:
                    save(model.module, os.path.join(args.save, 'best', 'best_test_model.pt'))
                else:
                    save(model, os.path.join(args.save, 'best', 'best_test_model.pt'))

                best_test_genotype_path = os.path.join(args.save, 'best', 'best_test_genotype.pkl')
                save_pickle(best_test_genotype, best_test_genotype_path)

        ## Get all operations from genotype
                
        assert len(genotype.edges) % num_keep_edges == 0
        nsteps = len(genotype.edges) // num_keep_edges        
        all_operation_name_cell_wise = []
        for i in range(nsteps):
            step_gene = genotype.steps[i]
            operation_name = []
            for j in range(node_steps):
                operation_name.append(step_gene.inner_steps[j])

                all_op_count[step_gene.inner_steps[j]]+=1
                if i == 0:
                    all_op_c1_count[step_gene.inner_steps[j]]+=1
                else:
                    all_op_c2_count[step_gene.inner_steps[j]]+=1
            all_operation_name_cell_wise.append(operation_name)
        #print("All operation name for epoch:",epoch," is: ",all_operation_name_cell_wise)
        file_name = "epoch_{}".format(epoch)
        file_name = os.path.join(args.save, "architectures", file_name)
        print("File name to store the plot is :",file_name)
        if(plotter != None):
            plotter.plot(genotype, file_name)
        print("1111111111111111111111111111111111111111111111111111111111")
        print("File name to store the plot is :",file_name)
        print("1111111111111111111111111111111111111111111111111111111111")
        # update the best model snap-shot after each epoch
        # model.load_state_dict(best_model_sd)
        # model.train(False)
        print("---------------------------------------------")
        print("predicted labels INdex: ")
        for i in pred_ind:
            print(i,end=" ")
        print()
        # print("predicted labels : ")
        # for i in pred_labels:
        #     print(i,end=" ")
        # print()
        # print("actual labels:")
        # for i in actual_labels:
        #     print(i,end=" ")
        # print()
        print("actual labels data:")
        for i in range(len(labels_data)):
            labels_data[i] = labels_data[i].item()
            print(labels_data[i],end=" ")
        print()
        pred_ind = np.array(pred_ind)
        labels_data = np.array(labels_data)
        # fpr, tpr, thresholds = metrics.roc_curve(labels_data, pred_ind, pos_label=2)
        # auc_values.append(metrics.auc(fpr, tpr))
        # print("Auc for this epoch:",auc_values[-1])

        logger.info("Current best dev accuracy: {}, at training epoch: {}".format(best_acc, best_epoch) )
        # if status is not 'search':
        logger.info("Current best test accuracy: {}, at training epoch: {}".format(best_test_acc, best_test_epoch) )
    collection_of_model = list(zip(all_acc, all_genotype,all_epoch,all_model))
    collection_of_model = sorted(collection_of_model, key = lambda x:x[0])
    n = len(collection_of_model)
    
    # writing all operation count in a file
    # print("All epoch operations:",all_op_count)
    # print("All cell 1 operation count:", all_op_c1_count)
    # print("All cell 2 operation count:", all_op_c2_count)

    all_operation_file_path = operation_file_path+"/op_count.txt"
    fptr = open(all_operation_file_path,"w")
    for key, val in all_op_count.items():
        temp = str(val)+"\n"
        fptr.write(temp)
    fptr.close()

    all_operation_cell1_file_path = operation_file_path+"/op_cell1_count.txt"
    fptr = open(all_operation_cell1_file_path,"w")
    for key, val in all_op_c1_count.items():
        temp = str(val)+"\n"
        fptr.write(temp)
    fptr.close()

    all_operation_cell2_file_path = operation_file_path+"/op_cell2_count.txt"
    fptr = open(all_operation_cell2_file_path,"w")
    for key, val in all_op_c2_count.items():
        temp = str(val)+"\n"
        fptr.write(temp)
    fptr.close()

    #Model storing code
    print("****Total No of geneotype: ", n , "*******")
    print("################### All accuracy of all epoch along with model ###################################")
    if(len(collection_of_model)!=0 and len(collection_of_model)>=3):
        for i in range(len(collection_of_model)-1,len(collection_of_model)-4,-1):
            print("Genotype no with epoch: ",collection_of_model[i][2])
            print(collection_of_model[i][1])
            print("Accuracy of genotype ", all_epoch[i]," is: ",collection_of_model[i][0])
            train_genotype_path = train_model_save+"/epoch_"+str(collection_of_model[i][2])+".pkl"
            print("Train_genotype_path:",train_genotype_path)
            save_pickle(collection_of_model[i][1], train_genotype_path)
            train_model_path = train_model_save+"/epoch_"+str(collection_of_model[i][2])+".pt"
            save(collection_of_model[i][3],train_model_path)

    # writing all the losses into a file
    train_file = open(train_loss_path, "w")
    val_file = open(dev_loss_path,"w")
    test_file = open(test_loss_path,"w")
    for i in range(len(train_loss)):
        s = str(train_loss[i])+"\n"
        train_file.write(s)
    for i in range(len(val_loss)):
        s = str(val_loss[i])+"\n"
        val_file.write(s)
    for i in range(len(test_loss)):
        s = str(test_loss[i])+"\n"
        test_file.write(s)
    print("Train_loss",train_loss)
    print("val_loss",val_loss)
    print("test_loss",test_loss)

    # Final layer values
    print("Final Layer Values:",len(final_layer_val))
    for i in range(len(final_layer_val)):
        print(final_layer_val[i])

    # writing the final layer values into file
    fptr = open(auc_data_path, "w")
    for i in range(len(final_layer_val)):
        s = ""
        for j in range(len(final_layer_val[i])):
            s+= str(final_layer_val[i][j])+" "
        s = s[:len(s)-1]
        s+= "\n"
        fptr.write(s)
    fptr.close()

    print("Alpha entropies: ",alpha_entropy_list)
    print("Gamma entropies: ",gamma_entropy_list)
    ####################### Store Entropy in file ####################################

    # alpha_path = entropy_file_path+"/alpha_val.txt"
    # fptr = open(alpha_path,'w')
    # for i in range(len(alpha_entropy_list)):
    #     s = str(alpha_entropy_list[i])
    #     s+="\n"
    #     fptr.write(s)
    # fptr.close()

    # gamma_path = entropy_file_path+"/gamma_val.txt"
    # fptr = open(gamma_path,'w')
    # for i in range(len(gamma_entropy_list)):
    #     s = str(gamma_entropy_list[i])
    #     s+="\n"
    #     fptr.write(s)
    # fptr.close()
    # print("Entropy data store at location: ",entropy_file_path)
    #####################################################################################

    
    if status == 'search':
        return best_acc, best_genotype
    else:
        return best_test_acc, best_genotype

def test_ntu_track_acc(model, dataloaders, criterion, genotype, 
                        dataset_sizes, device, logger, args):
    model.eval()
    # Each epoch has a training and validation phase
    logger.info("EXP: {}".format(args.save) )
    phase = 'test'

    running_loss = 0.0
    running_corrects = 0
    pred_labels = []
    actual_labels = []
    with tqdm(dataloaders[phase]) as t:
        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            rgbs, spec, labels = data['rgb'], data['spec'], data['label']
            # device
            rgbs = rgbs.to(device)
            spec = spec.to(device)
            labels = labels.to(device)
            actual_labels.append(labels)
            # continue
            input_features = (rgbs, spec)
            # zero the parameter gradients
            
            output = model(input_features)
            _, preds = torch.max(output, 1)
            pred_labels.append(preds)
            loss = criterion(output, labels)

            # statistics
            running_loss += loss.item() * rgbs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            batch_acc = torch.sum(preds == labels.data) * 1.0 / rgbs.size(0) 
            postfix_str = 'batch_loss: {:.03f}, batch_acc: {:.03f}'.format(loss.item(), batch_acc)

            t.set_postfix_str(postfix_str)
            t.update()

    test_loss = running_loss / dataset_sizes[phase]
    test_acc  = running_corrects.double() / dataset_sizes[phase]
    
    logger.info(str(genotype))
    logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, test_loss, test_acc))
   

    return test_acc
