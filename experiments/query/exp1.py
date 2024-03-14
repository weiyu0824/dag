from nuimages import NuImages
import tqdm
import cv2
import numpy as np
from ultralytics import YOLO 
import torch
from metric import calculate_map

def draw(img_path, bboxes, save_path):
    img = cv2.imread(img_path)
    color = (0, 255, 0)  # BGR color format (green)
    thickness = 2 
    for bbox in bboxes:
        # print('cool', bbox)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness) 
    cv2.imwrite(save_path, img)



# Load data
root = 'nuscenes-devkit/data/sets/nuimages/'
nuim = NuImages(dataroot=root, version='v1.0-mini', verbose=False, lazy=True)
num_frame = len(nuim.sample)

# Load model
model_name = 'yolov8l'
model = YOLO(f'{model_name}.pt')
model.info()


# metric
pre_latency, inf_latency, post_latency = [], [], []
for i in tqdm.tqdm(range(num_frame)):
    sample = nuim.sample[i]
    key_camera_token = sample['key_camera_token']

    # Get object annos & category
    bboxes, categories = [], []
    for obj_ann in nuim.object_ann:
        if obj_ann['sample_data_token'] == 'key_camera_token':
            bboxes.append(obj_ann['bbox'])
            categories.append(nuim.get('category', obj_ann['category_token']))

    # Get filename
    filename = nuim.get('sample_data', sample['key_camera_token'])['filename']

    # inference
    import time 
    st = time.time()
    # print(st)
    results = model.predict(root + filename, save=False)
    result = results[0]
    # pred_bboxes = results[0].boxes.xyxyn.numpy()   
    # print(results)
    # print(time.time() - st)

    # calculate latency 
    pre_latency.append(result.speed['preprocess'])
    inf_latency.append(result.speed['inference'])
    post_latency.append(result.speed['postprocess'])

    # print(result.conf)
    pred_clss = result.boxes.cls
    pred_confs = result.boxes.conf
    pred_bboxes = result.boxes.xyxy.round().int()
    
    # pred_boxes = 
    attr = torch.stack([pred_clss, pred_confs]).T
    pred_boxes = torch.cat([pred_bboxes, attr], dim=1 )
    print(pred_boxes)

    # gt boxes=
     

    # calculate mAP
    # calculate_map(true_boxes=, pred_boxes=)


print(sum(pre_latency)/num_frame)
print(sum(inf_latency)/num_frame)
print(sum(post_latency)/num_frame)
exit()

key_camera_token = sample['key_camera_token']
# object_tokens, surface_tokens = nuim.list_anns(sample['token'])
bboxes = [o['category_token'] for o in nuim.object_ann if o['sample_data_token'] == key_camera_token]
print(bboxes)
print(nuim.get('category', bboxes[0]))
exit()

img_path = root + nuim.get('sample_data', sample['key_camera_token'])['filename']

# draw(img_path, bboxes, 'test-1.jpg')
# img_path = '/Users/andersonlin/Documents/research/chain-optimize/nuscenes-devkit/data/sets/nuimages/samples/CAM_FRONT_RIGHT/n013-2018-08-27-14-41-26+0800__CAM_FRONT_RIGHT__1535352274870176.jpg'

# # key_camera_token = sample['key_camera_token']
# # nuim.render_image(key_camera_token, object_tokens=object_tokens)

from ultralytics import YOLO 
print('Load')
model_name = 'yolov8x'
model = YOLO(f'{model_name}.pt')
print('Info')
model.info()
# print('Train')
# results = model.train(data='coco8.yaml', epochs=100, imgsz=640)
print('Inf')
results = model.predict(img_path, save=False)
print(results[0].boxes)

exit()
org_h, org_w = results[0].orig_shape
# print(results[0].boxes.xywh.numpy())
# print(results[0].boxes.xyxy.numpy())
print(results[0].boxes.xyxy.numpy())
print(results[0].boxes.xyxyn.numpy())

bboxes = results[0].boxes.xyxyn.numpy()

bboxes[:, 0] = bboxes[:, 0] * org_w
bboxes[:, 1] = bboxes[:, 1] * org_h
bboxes[:, 2] = bboxes[:, 2] * org_w
bboxes[:, 3] = bboxes[:, 3] * org_h

# # print(results)

# bboxes = results[0].boxes.xywh.numpy()
# bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] 
# bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
bboxes = np.round(bboxes).astype(np.int)
print(bboxes)

# draw(img_path, bboxes, f'pred-{model_name}.jpg')


# import yolov5
# for model_name in ['keremberke/yolov5n-aerial-sheep', 'keremberke/yolov5s-aerial-sheep', 'keremberke/yolov5m-aerial-sheep']:
#     model = yolov5.load(model_name)
#     print('Loaded')
#     model.conf = 0.25  # NMS confidence threshold
#     model.iou = 0.45  # NMS IoU threshold
#     model.agnostic = False  # NMS class-agnostic
#     model.multi_label = False  # NMS multiple labels per box
#     model.max_det = 1000  # maximum number of detections per image
#     print('Perform')
#     # perform inference
#     preprocess_times, inference_times, postprocess_times = [], [], []
#     num_frame = 1
#     for sample_idx in tqdm.tqdm(range(num_frame)):
#         img_path = root + nuim.sample_data[sample_idx]['filename']
#         print(img_path)
#         results = model(img_path)
#         preprocess_times.append(results.times[0].t)
#         inference_times.append(results.times[1].t)
#         postprocess_times.append(results.times[2].t)

#         # parse results
#         predictions = results.pred[0]
#         boxes = predictions[:, :4] # x1, y1, x2, y2
#         scores = predictions[:, 4]
#         categories = predictions[:, 5]
#         print(boxes, scores, categories)
        
#         # Calculate Acc
#         # mAp:

#     break
#     print(f'Avg speed {model_name}: \
#             pre-process: {sum(preprocess_times)/num_frame}, \
#             inference: {sum(inference_times)/num_frame}')