
import argparse
import torch
from tqdm import tqdm
from pathlib import Path
import shutil, os, sys
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



#sys.path.append("/home/ysgnagender/NER/IE/content/IE/CRAFT/")
#sys.path.insert(0, "/home/ysgnagender/NER/IE/content/IE/CRAFT/")
#sys.path.append("../../OCR/Scatter/")
#dir = '/home/ysgnagender/NER/IE/OCR/CRAFT/'
#import sample

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def main(args):
    #tmp()
    #os.system('python3 /home/ysgnagender/NER/IE/content/IE/CRAFT/test.py')
    #execfile('sample.py')
    #sample.main()

    #best_model = 'saved/models/PICK_Default/pos_transformation/checkpoint-epoch100.pth'
    device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']
    print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(args.checkpoint, monitor_best))
    
    #bt = '../CSV/'
    
    #nam = os.path.basename(args.im)
    #nam = args.im
    #print(args.im, nam, bt)
    #shutil.copy('boxes_and_transcripts_no_lab/' + nam.replace('.jpeg', '.csv'), bt)

    # prepare model for testing
    pick_model = config.init_obj('model_arch', pick_arch_module)
    pick_model = pick_model.to(device)
    pick_model.load_state_dict(state_dict)
    pick_model.eval()

    # setup dataset and data_loader instances
    test_dataset = PICKDataset(boxes_and_transcripts_folder=args.bt,
                               images_folder=args.impt,
                               resized_image_size=(480, 960),
                               ignore_error=False,
                               training=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                  num_workers=2, collate_fn=BatchCollateFn(training=False))

    # setup output path
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
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

            BB_path = 'data/test_data_example/boxes_and_transcripts/'
            #BB_path = '../Scatter/TSV_GI/'
            bb = input_data_item['boxes_coordinate_org']
            #print('BB', bb)
            entity_nam = ['Company_Name', 'Address', 'Date', 'Item_Name', 'Item_Price', 'Total', 'Sub_Total', 'Quantity', 'Phone_Number', 'Time', 'Tax', 'Ind_Item_Price']
            
            with open(BB_path + file_name.replace('.jpeg', '.tsv'), "r") as fp:
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

                            if df.loc[j,10] == 'Company_Name':
                                node_stat_end[i,2] = 1                       
                            elif df.loc[j,10] == 'Address':
                                node_stat_end[i,2] = 2
                            elif df.loc[j,10] == 'Date':
                                node_stat_end[i,2] = 3
                            elif df.loc[j,10] == 'Item_Name':
                                node_stat_end[i,2] = 4
                            elif df.loc[j,10] == 'Item_Price':
                                node_stat_end[i,2] = 5
                            elif df.loc[j,10] == 'Total':
                                node_stat_end[i,2] = 6
                            elif df.loc[j,10] == 'Sub_Total':
                                node_stat_end[i,2] = 7
                            elif df.loc[j,10] == 'Quantity':
                                node_stat_end[i,2] = 8
                            elif df.loc[j,10] == 'Phone_Number':
                                node_stat_end[i,2] = 9
                            elif df.loc[j,10] == 'Time':
                                node_stat_end[i,2] = 10
                            elif df.loc[j,10] == 'Tax':
                                node_stat_end[i,2] = 11
                            elif df.loc[j,10] == 'Ind_Item_Price':
                                node_stat_end[i,2] = 12

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
                
                #print('+++++++++++++++++\n', bb[0,0,0], bb[0,0,2])

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

                #print('-----------------------Span\n', spans)        
                #print(len(df.index), len(spans))
                
                for i in range(len(df.index)):
                    for j in range(len(spans)):
                        iv = pandas.Interval(left=int(node_stat_end[i,0]), right=int(node_stat_end[i,1]), closed='both')
                        if (pred[j,0] in iv) and (pred[j,1] in iv):
                            node_stat_end[i,3] = pred[j,2]
                            break

                #print('F1 Score: Macro', f1_score(node_stat_end[:,2], node_stat_end[:,3], average='macro'), 'Micro', f1_score(node_stat_end[:,2], node_stat_end[:,3], average='micro'))
                #print(f1_score(node_stat_end[:,2], node_stat_end[:,3], average='micro'))
                
                F1_Score_macro.append(f1_score(node_stat_end[:,2], node_stat_end[:,3], average='macro'))
                F1_Score_micro.append(f1_score(node_stat_end[:,2], node_stat_end[:,3], average='micro'))

                '''
                for i in range(len(spans)):
                    for j in range(len(df.index)):
                        iv = pandas.Interval(left=int(node_stat_end[j,0]), right=int(node_stat_end[j,1]), closed='both')
                        if (pred[i,0] in iv) and (pred[i,1] in iv):
                            pred[i,3] = node_stat_end[j,2]
                            break

                print('F1 Score', f1_score(pred[:,3], pred[:,2], average='macro'))
                print(f1_score(pred[:,3], pred[:,2], average='micro'))
                
                F1_Score_macro.append(f1_score(pred[:,3], pred[:,2], average='macro'))
                F1_Score_micro.append(f1_score(pred[:,3], pred[:,2], average='micro'))
                '''

                for entity_name, range_tuple in spans:
                    entity = dict(entity_name=entity_name,
                                  text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)
                    #print('entity', entity)

                result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
                
                with result_file.open(mode='w') as f:
                    for item in entities:
                        f.write('{}\t{}\n'.format(item['entity_name'], item['text']))            
                
                entity_ind_ord = [1, 2, 3, 10, 9, 4, 8, 12, 5, 6, 7, 11]

                for ii in entity_ind_ord:
                    #print('-----------', entity_nam[ii-1], '------------')
                    ind =list(locate(pred[:,2], lambda x: x == ii))
                    if len(ind) != 0:
                        #print('Indices', ind)
                        '''
                        min_ind = list(pred[ind,3]).index(torch.min(pred[ind,3]))
                        #tmp = np.array(pred[ind,3])
                        #min_ind = np.where(tmp == np.amin(tmp))

                        #print('ind', ind, pred[ind,3], min_ind, pred[min_ind,3])
                
                        
                        #ind_ot = [x for x in ind if x != min_ind]
                        angle = []
                        for i in ind:
                            angle.append(angle_between(np.array(pred[min_ind,3:5]), np.array(pred[i,3:5])))
                        #print(angle)
                        sort_ind = np.argsort(angle)
                        #sort_ind = np.insert(sort_ind, 0, min_ind)
                        #print('sorted', sort_ind)
                        sorted_text = ''
                        for i in sort_ind:
                            sorted_text = sorted_text + ' ' + ''.join(decoded_texts[int(pred[i,0]):int(pred[i,1]+1)])
                        print(entity_nam[ii], ': ', sorted_text)
                        '''

                        
                        '''
                        print(pred[ind,3:11])
                        for i in ind:                           
                            print(''.join(decoded_texts[int(pred[i,0]):int(pred[i,1]+1)]))
                        '''
                        ext_ent = ''
                        while len(ind) > 0:
                            tot_list = []
                            ymin_ind = list(pred[ind,4]).index(torch.min(pred[ind,4]))
                            y_ind = []
                            for kk in ind:
                                if (pred[kk,4] >= pred[ind[ymin_ind],4]) and (pred[kk,4] <= pred[ind[ymin_ind],4]+(word_ht_mean*2/3)):
                                    y_ind.append(kk) 
                                    tot_list.append(kk)
                            #print('Line ind', ind, y_ind, 'ymin_ind', ymin_ind, word_ht_mean)

                            xmin_ind = list(pred[y_ind,3]).index(torch.min(pred[y_ind,3]))
                            #print('Left most word:', ''.join(decoded_texts[int(pred[y_ind[xmin_ind],0]):int(pred[y_ind[xmin_ind],1]+1)]))
                            angle = []
                            for i in y_ind:
                                angle.append(angle_between(np.array(pred[y_ind[xmin_ind],3:5]), np.array(pred[i,3:5])))
                            #print(angle)
                            sort_ind = np.argsort(angle)
                            #sort_ind = np.insert(sort_ind, 0, min_ind)
                            #print('sorted', sort_ind)
                            
                            sorted_text = ''
                            for i in sort_ind:
                                sorted_text = sorted_text + ' ' + ''.join(decoded_texts[int(pred[y_ind[i],0]):int(pred[y_ind[i],1]+1)])
                            #print(entity_nam[ii], ': -----------------------------------', sorted_text)
                            ind = list(set(ind) - set(tot_list))
                            if ii != 4:
                                ext_ent = ext_ent + ' ' + sorted_text
                            else:
                                ext_ent = ext_ent + '\n ' + sorted_text

                        #print(entity_nam[ii-1], ': ', ext_ent)

                        #pred[min_ind,4] word_ht_mean



                '''
                # Writing labels to a file
                print('--', file_name.replace('.jpg', '.txt'))
                with open(label_path + file_name.replace('.jpg', '.txt'), 'w') as f1:
                    for lab in node_stat_end:
                        lab = lab.numpy()
                        #print(lab.shape)
                        f1.write(str(int(lab[3])))
                        f1.write('\n')
                f1.close()
                '''
            #print('----------------------------------------------------- \n')
        
    print('Average F1 Score Macro:', sum(F1_Score_macro)/len(F1_Score_macro), 'Micro:', sum(F1_Score_micro)/len(F1_Score_micro))

if __name__ == '__main__':
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
    main(args)
