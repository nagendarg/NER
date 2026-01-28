
import argparse
import torch
from tqdm import tqdm
from pathlib import Path
import shutil, os, sys
import csv, pandas


from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from parse_config import ConfigParser
import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str
from sklearn.metrics import f1_score


#sys.path.append("/home/ysgnagender/NER/IE/content/IE/CRAFT/")
#sys.path.insert(0, "/home/ysgnagender/NER/IE/content/IE/CRAFT/")
#sys.path.append("../../OCR/Scatter/")
#dir = '/home/ysgnagender/NER/IE/OCR/CRAFT/'
#import sample

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
    #shutil.copy('boxes_and_transcripts_no_lab/' + nam.replace('.jpg', '.csv'), bt)

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
            with open(BB_path + file_name.replace('.jpg', '.tsv'), "r") as fp:
                reader = csv.reader(fp, delimiter=",")
                data_read = [row for row in reader]              
                df = pandas.DataFrame(data_read)

                node_stat_end = torch.zeros(len(df.index), 4)
                for i in range(len(df.index)):
                    if df.loc[i,10] == 'Company_Name':
                        node_stat_end[i,2] = 1                       
                    if df.loc[i,10] == 'Address':
                        node_stat_end[i,2] = 2
                    if df.loc[i,10] == 'Date':
                        node_stat_end[i,2] = 3
                    if df.loc[i,10] == 'Item_Name':
                        node_stat_end[i,2] = 4
                    if df.loc[i,10] == 'Item_Price':
                        node_stat_end[i,2] = 5
                    if df.loc[i,10] == 'Total':
                        node_stat_end[i,2] = 6

                    if i == 0:
                        #print('0', len(df.loc[i,9]))
                        node_stat_end[i,0] = 0
                        node_stat_end[i,1] = len(df.loc[i,9]) - 1
                    else:
                        #print(len(df.loc[i-1,9]), len(df.loc[i,9]))
                        node_stat_end[i,0] = node_stat_end[i-1,1] + 1
                        node_stat_end[i,1] = node_stat_end[i,0] + len(df.loc[i,9]) - 1

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

                entities = []  # exists one to many case

                pred = 0*torch.ones(len(spans), 3)

                
                for i in range(len(spans)):
                    pred[i,0] = spans[i][1][0]
                    pred[i,1] = spans[i][1][1]

                    if spans[i][0] == 'Company_Name':
                        pred[i,2] = 1
                        
                    if spans[i][0] == 'Address':
                        pred[i,2] = 2
                        
                    if spans[i][0] == 'Date':
                        pred[i,2] = 3
                        
                    if spans[i][0] == 'Item_Name':
                        pred[i,2] = 4 
                        
                    if spans[i][0] == 'Item_Price':
                        pred[i,2] = 5 
                        
                    if spans[i][0] == 'Total':
                        pred[i,2] = 6
                        
                print('entities length', len(df.index), len(spans))
                for i in range(len(df.index)):
                    for j in range(len(spans)):
                        iv = pandas.Interval(left=int(node_stat_end[i,0]), right=int(node_stat_end[i,1]), closed='both')
                        #print(int(pred[j,0]), int(pred[j,1]), '----', int(node_stat_end[i,0]), int(node_stat_end[i,1]))
                        if (pred[j,0] in iv) and (pred[j,1] in iv):
                            node_stat_end[i,3] = pred[j,2]
                            break
                            
                print('F1 Score', f1_score(node_stat_end[:,2], node_stat_end[:,3], average='macro'))
                print(f1_score(node_stat_end[:,2], node_stat_end[:,3], average='micro'))
                #print(node_stat_end)
                F1_Score_macro.append(f1_score(node_stat_end[:,2], node_stat_end[:,3], average='macro'))
                F1_Score_micro.append(f1_score(node_stat_end[:,2], node_stat_end[:,3], average='micro'))

                for entity_name, range_tuple in spans:
                    for i in range(len(df.index)):
                        iv = pandas.Interval(left=int(node_stat_end[i,0]), right=int(node_stat_end[i,1]), closed='both')
                        if (range_tuple[0] in iv) and (range_tuple[1] in iv):
                            stat = int(node_stat_end[i,0])
                            end_p = int(node_stat_end[i,1])
                            print('---', range_tuple[0], range_tuple[1], '--->', stat, end_p)
                            break

                    #entity = dict(entity_name=entity_name,
                    #              text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))

                    entity = dict(entity_name=entity_name,
                                  text=''.join(decoded_texts[stat:end_p + 1]))
                    entities.append(entity)
                    #print('entity', entity)

                result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
                
                with result_file.open(mode='w') as f:
                    for item in entities:
                        f.write('{}\t{}\n'.format(item['entity_name'], item['text']))            
                
                '''
                print('--', file_name.replace('.jpg', '.txt'))
                # Writing predicted labels to a file
                with open(label_path + file_name.replace('.jpg', '.txt'), 'w') as f1:
                    for lab in node_stat_end:
                        lab = lab.numpy()
                        #print(lab.shape)
                        f1.write(str(int(lab[3])))
                        f1.write('\n')
                f1.close()
                '''


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
