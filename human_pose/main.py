from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

import numpy as np
import PIL
import yaml
import os
import torch
import sys

from tqdm import tqdm

import dataset as human_dataset


def get_info(config, mode='test'):
    dataset_param = config['dataset_params']
    train_param = config['train_params']

    dataset = getattr(human_dataset,dataset_param['dataset']['name'])\
                                    (dataset_param['dataset']['{}_image_set'.format(mode)],
                                        dataset_param['dataset']['path'],
                                    train_param['patch_width'], train_param['patch_height'],
                                    train_param['use_philly'],
                                    train_param['rect_3d_width'], train_param['rect_3d_height'],
                                    dataset_param['dataset']['extra_param'], init_mode=False)
    folder, start, end = dataset.get_folder_info()

    return folder, start, end, dataset

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

def prepare_keypoints(keypoints, transform, device, shape):
    keypoints = transform.apply_coords(keypoints, shape)
    keypoints = torch.as_tensor(keypoints, device=device.device)
    return keypoints

def segment_human(sam, resize_transform, gt_db):

    batched_input = []
    for cam in range(0, 4):
        cam_key = 'cam_{}'.format(cam)

        image_path = gt_db[cam_key]['image']
        keypoints = gt_db[cam_key]['joints_3d'][:, :2]

        image = PIL.Image.open(image_path)
        image = np.array(image)

        keypoints_tensor = prepare_keypoints(keypoints, resize_transform, sam, shape=(image.shape[0], image.shape[1]))
        keypoints_label = torch.ones(keypoints_tensor.shape[0]).to(sam.device)

        keypoints_tensor = keypoints_tensor.unsqueeze(0)
        keypoints_label = keypoints_label.unsqueeze(0)
        
        image_tensor = prepare_image(image, resize_transform, sam)
 
        batched_input.append({
            'image': image_tensor,
            'point_coords': keypoints_tensor,
            'point_labels': keypoints_label,
            'original_size': image.shape[:2]})

    batched_output = sam(batched_input, multimask_output=False)

    return batched_output

def writer(gt_db, batched_output):
    output_path = 'background_sam'

    for cam in range(0, 4):
        cam_key = 'cam_{}'.format(cam)
        image_path = gt_db[cam_key]['image']

        output_folder, output_name = image_path.split('data/hm36/images/')[-1].split('/')

        output_name = output_name.replace('.jpg', '.png')

        if not os.path.exists(os.path.join(output_path, output_folder)):
            os.makedirs(os.path.join(output_path, output_folder))

        # write mask to file
        mask = batched_output[cam]['masks'].squeeze().cpu().numpy()
        mask = PIL.Image.fromarray(mask)
        mask.save(os.path.join(output_path, output_folder, output_name))

        # img = PIL.Image.open(image_path)
        # img.save(os.path.join(output_path, output_folder, output_name.replace('.png', '_img.png')))


if __name__ == '__main__':
    config = yaml.load(open('human_pose/config.yaml', 'r'), Loader=yaml.FullLoader)

    sam_vit_model = './vit_model/sam_vit_l_0b3195.pth'
    model_type = 'vit_l'

    # sam_vit_model = './vit_model/sam_vit_b_01ec64.pth'
    # model_type = 'vit_b'
    
    # get argv
    argv = sys.argv
    assert argv[1] in ['test', 'train']

    folder, start, end, dataset = get_info(config, mode=argv[1])

    device='cuda'

    sam = sam_model_registry[model_type](checkpoint=sam_vit_model)
    sam.to(device=device)

    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    for i in tqdm(range(start, end), desc='Folder'):
        gt_dbs = dataset.gt_db(folder[i])
        
        with tqdm(total=len(gt_dbs), desc='Image') as pbar:
            for gt_db in gt_dbs:
                output = segment_human(sam, resize_transform, gt_db)
                writer(gt_db, output)
                pbar.update(1)
        
            pbar.leave = False
