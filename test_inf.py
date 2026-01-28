
import argparse
import torch
from tqdm import tqdm
from pathlib import Path
import shutil, os, sys, json, time
import csv, pandas, math
import numpy as np


from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from parse_config import ConfigParser
import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str
from sklearn.metrics import f1_score
from more_itertools import locate


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def main_c():

    best_model = '/var/tellme/users/nagendar/DBNet/NER/trained_model/checkpoint-epoch70.pth'
    #gpu = -1
    #device = torch.device(f'cuda:{gpu}' if gpu != -1 else 'cpu')
    #torch.cuda.set_device(0)
    #device = 'cuda'
    device = torch.device('cuda:0')
    start = time.time()
    checkpoint = torch.load(best_model, map_location=device)
    end = time.time()
    print("NER model loading took {:.6f} seconds".format(end - start))

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']

    # prepare model for testing
    pick_model = config.init_obj('model_arch', pick_arch_module)
    pick_model = pick_model.to(device)
    pick_model.load_state_dict(state_dict)
    pick_model.eval()

    # setup dataset and data_loader instances
    test_dataset = PICKDataset(boxes_and_transcripts_folder='/var/tellme/users/nagendar/BB/',
                               images_folder='/var/tellme/users/nagendar/DBNet/test_images_1/',
                               resized_image_size=(480, 960),
                               ignore_error=False,
                               training=False)
    #test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
    #                              num_workers=1, collate_fn=BatchCollateFn(training=False))

    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                  num_workers=0, collate_fn=BatchCollateFn(training=False))
    # setup output path
    #output_path = Path(args.output_folder)
    #output_path.mkdir(parents=True, exist_ok=True)
    label_path = 'labels/'

    F1_Score_macro = []
    F1_Score_micro = []
    # predict and save to file
    with torch.no_grad():
        for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
            for key, input_value in input_data_item.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    input_data_item[key] = input_value.to(device)
  
            # For easier debug.
            #image_names = input_data_item["filenames"]
            print(os.path.basename(input_data_item["filenames"][0]))
            #print('\n', input_data_item["filenames"][0])
            file_name = os.path.basename(input_data_item["filenames"][0])
            #print(input_data_item["filenames"])
            #print('---', file_name.replace('.jpeg', '.txt'))

            BB_path = '/var/tellme/users/nagendar/BB/'
            #BB_path = '../Scatter/TSV_GI/'
            bb = input_data_item['boxes_coordinate_org']
            #print('BB', bb)
            entity_nam = ['Company_Name', 'Address', 'Date', 'Item_Name', 'Item_Price', 'Total', 'Sub_Total', 'Quantity', 'Phone_Number', 'Time', 'Tax', 'Ind_Item_Price']
            
            #df_1 = pandas.read_csv(BB_path + file_name.replace('.jpg', '.tsv'), delimiter=',')
            #print('df_1', df_1)
            with open(BB_path + file_name.replace('.jpg', '.tsv'), "r") as fp:
                reader = csv.reader(fp, delimiter=",")
                data_read = [row for row in reader]              
                df = pandas.DataFrame(data_read)
                tot_box = len(df.index)
                word_ht_mean = np.mean(df[8].astype(str).astype(int)-df[2].astype(str).astype(int))

                
                node_stat_end = torch.zeros(len(df.index), 4)

                for i in range(len(df.index)):
                    for j in range(len(df.index)):

                        if (int(bb[0,i,0])==int(df.loc[j,1])) and (int(bb[0,i,1])==int(df.loc[j,2])):
                            if i == 0:
                                #print('0', len(df.loc[i,9]))
                                node_stat_end[i,0] = 0
                                node_stat_end[i,1] = len(df.loc[j,9]) - 1
                            else:
                                #print(len(df.loc[i-1,9]), len(df.loc[i,9]))
                                node_stat_end[i,0] = node_stat_end[i-1,1] + 1
                                node_stat_end[i,1] = node_stat_end[i,0] + len(df.loc[j,9]) - 1
                            #print(bb[0,i,0], df.loc[j,1], bb[0,i,2], df.loc[j,3])

                            break
                

            output = pick_model(**input_data_item)
            #print('output', output)

            logits = output['logits']  # (B, N*T, out_dim)
            new_mask = output['new_mask']
            image_indexs = input_data_item['image_indexs']  # (B,)
            text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
            mask = input_data_item['mask']
            # List[(List[int], torch.Tensor)]
            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)
            #print('predicted tags', predicted_tags)

            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)
            #print('decoded text', decoded_texts_list)

            for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                # List[ Tuple[str, Tuple[int, int]] ]
                spans = bio_tags_to_spans(decoded_tags, [])
                spans = sorted(spans, key=lambda x: x[1][0])
                #print('====', input_data_item['boxes_coordinate_org'])

                entities = []  # exists one to many case

                total_text = ''.join(decoded_texts[:-1])
                
                pred = 0*torch.ones(len(spans), 11)
                
                for i in range(len(spans)):

                    pred[i,0] = spans[i][1][0]
                    pred[i,1] = spans[i][1][1]

                    for jj in range(tot_box):
                        if (pred[i,0]==node_stat_end[jj,0]) and (pred[i,1]==node_stat_end[jj,1]):
                            pred[i,3:11] = bb[0,jj,0:8]

                    if spans[i][0] == 'Company_Name':
                        pred[i,2] = 1                       
                    elif spans[i][0] == 'Address':
                        pred[i,2] = 2                        
                    elif spans[i][0] == 'Date':
                        pred[i,2] = 3                        
                    elif spans[i][0] == 'Item_Name':
                        pred[i,2] = 4                         
                    elif spans[i][0] == 'Item_Price':
                        pred[i,2] = 5                         
                    elif spans[i][0] == 'Total':
                        pred[i,2] = 6
                    elif spans[i][0] == 'Sub_Total':
                        pred[i,2] = 7
                    elif spans[i][0] == 'Quantity':
                        pred[i,2] = 8
                    elif spans[i][0] == 'Phone_Number':
                        pred[i,2] = 9
                    elif spans[i][0] == 'Time':
                        pred[i,2] = 10
                    elif spans[i][0] == 'Tax':
                        pred[i,2] = 11
                    elif spans[i][0] == 'Ind_Item_Price':
                        pred[i,2] = 12            
    
                #entity_nam = ['Company_Name', 'Address', 'Date', 'Item_Name', 'Item_Price', 'Total', 'Sub_Total', 'Quantity', 'Phone_Number', 'Time', 'Tax', 'Ind_Item_Price']
                entity_ind_ord = [2, 1, 3, 10, 9, 4, 8, 12, 5, 6, 7, 11]

                res_ent = []
                ent_d = '{}'
                ent_json = json.loads(ent_d)
                res_file = '../OP_Entities/entities.txt'
                count = 0
                with open(res_file, 'w') as f:
                    for ii in entity_ind_ord:
                        #print('-----------', entity_nam[ii-1], '------------')
                        ind =list(locate(pred[:,2], lambda x: x == ii))
                        if len(ind) != 0:
                            #print('Indices', ind)
                            count += 1
 
                            ext_ent = ''
                            count_1 = 0
                            while len(ind) > 0:
                                tot_list = []
                                ymin_ind = list(pred[ind,4]).index(torch.min(pred[ind,4]))
                                y_ind = []
                                for kk in ind:
                                    if (pred[kk,4] >= pred[ind[ymin_ind],4]) and (pred[kk,4] <= pred[ind[ymin_ind],4]+(word_ht_mean*2/3)):
                                        y_ind.append(kk) 
                                        tot_list.append(kk)
                                #print('Line ind', ind, y_ind, 'ymin_ind', ymin_ind, word_ht_mean)
                                #print('pred[y_ind,3]', pred[y_ind,3])

                                xmin_ind = list(pred[y_ind,3]).index(torch.min(pred[y_ind,3]))

                                #print('xmin_ind', xmin_ind, 'ind', ind)
                                #print('Left most word:', ''.join(decoded_texts[int(pred[y_ind[xmin_ind],0]):int(pred[y_ind[xmin_ind],1]+1)]))
                                angle = []
                                for i in y_ind:
                                    angle.append(angle_between(np.array(pred[y_ind[xmin_ind],3:5]), np.array(pred[i,3:5])))
                                #print('angle', angle)
                                sort_ind = np.argsort(angle)
                            
                                sorted_text = ''
                                for i in sort_ind:
                                    sorted_text = sorted_text + ' ' + ''.join(decoded_texts[int(pred[y_ind[i],0]):int(pred[y_ind[i],1]+1)])
                                #print(sorted_text)
                                ind = list(set(ind) - set(tot_list))
                                if count_1 == 0:
                                    ext_ent = ext_ent + sorted_text
                                else:
                                    ext_ent = ext_ent + ',' + sorted_text
                                count_1 += 1
                                '''
                                if ii != 4:
                                    ext_ent = ext_ent + ' ' + sorted_text
                                else:
                                    ext_ent = ext_ent + '\n ' + sorted_text
                                '''
                            #res_ent.append([entity_nam[ii-1], ext_ent])
                            #f.write('{}\t{}\n'.format(entity_nam[ii-1], ext_ent))
                            #res_ent.append([entity_nam[ii-1], ext_ent])
                            tmp = {entity_nam[ii-1]: ext_ent}
                            ent_json.update(tmp)
    json.dumps(ent_json)
    with open("entities.json", "w") as outfile:
        json.dump(ent_json, outfile)    

    #return res_ent

if __name__ == '__main__':
    start = time.time()

    args = argparse.ArgumentParser(description='PyTorch PICK Testing')
    args.add_argument('-ckpt', '--checkpoint', default=None, type=str,
                      help='path to load checkpoint (default: None)')
    args.add_argument('--bt', '--boxes_transcripts', default=None, type=str,
                      help='ocr results folder including boxes and transcripts (default: None)')
    args.add_argument('--impt', '--images_path', default=None, type=str,
                      help='images folder path (default: None)')
    args.add_argument('-output', '--output_folder', default='Extracted_Entities', type=str,
                      help='output folder (default: ../Extracted_Entities)')
    args.add_argument('-g', '--gpu', default=-1, type=int,
                      help='GPU id to use. (default: -1, cpu)')
    args.add_argument('--bs', '--batch_size', default=1, type=int,
                      help='batch size (default: 1)')
    args.add_argument('--im', '--image', default=None, type=str,
                      help='batch size (default: 1)')

    args = args.parse_args()
    main_c()
    end = time.time()
    print("Inside NER code took {:.6f} seconds".format(end - start))
