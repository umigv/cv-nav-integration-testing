from ultralytics import YOLO
import numpy as np
import time
import os
from PIL import Image
import cv2
import json
from tqdm import tqdm

def compute_mean_iou(ground_truth, prediction):
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def compute_dice_coefficient(ground_truth, prediction):
    intersection = np.logical_and(ground_truth, prediction)
    dice_coefficient = 2 * np.sum(intersection) / (np.sum(ground_truth) + np.sum(prediction))
    return dice_coefficient

def compute_pixel_accuracy(ground_truth, prediction):
    correct_pixels = np.sum(ground_truth == prediction)
    total_pixels = np.prod(ground_truth.shape)
    pixel_accuracy = correct_pixels / total_pixels
    return pixel_accuracy

def compute_metrics(ground_truth, prediction):
    iou_score = compute_mean_iou(ground_truth, prediction)
    dice_coefficient = compute_dice_coefficient(ground_truth, prediction)
    pixel_accuracy = compute_pixel_accuracy(ground_truth, prediction)
    return iou_score, dice_coefficient, pixel_accuracy

def polygon_to_mask(polygon, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    polygon = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [polygon], 1)
    return mask

def load_data():
    image_files = sorted(os.listdir('Drivable-area-model-8/valid/images'))
    label_files = sorted(os.listdir('Drivable-area-model-8/valid/labels'))

    for image_file, label_file in zip(image_files, label_files):
        image = Image.open(f'Drivable-area-model-8/valid/images/{image_file}')
        with open(f'Drivable-area-model-8/valid/labels/{label_file}', 'r') as f:
            label_data = f.read().split()
            label_data = [float(x) for x in label_data[1:]]
            label_data = [(int(label_data[i]*image.width), int(label_data[i+1]*image.height)) for i in range(0, len(label_data), 2)]
            width, height = image.size
            label = polygon_to_mask(label_data, (height, width))

        yield image, label

def visualize_and_save(image, ground_truth, prediction, frame_number, output_dir):

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    prediction_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    gt_mask[ground_truth == 1] = 255
    prediction_mask[prediction == 1] = 255

    gt_colored_mask = np.dstack((np.zeros_like(gt_mask), gt_mask, np.zeros_like(gt_mask)))
    prediction_colored_mask = np.dstack((np.zeros_like(prediction_mask), np.zeros_like(prediction_mask), prediction_mask))

    alpha = 0.5
    combined = image.copy()
    np.putmask(combined, gt_colored_mask.astype(bool), cv2.addWeighted(image, 1-alpha, gt_colored_mask, alpha, 0))
    np.putmask(combined, prediction_colored_mask.astype(bool), cv2.addWeighted(image, 1-alpha, prediction_colored_mask, alpha, 0))

    frame_filename = f"{output_dir}/frame_{frame_number:04d}.png"
    cv2.imwrite(frame_filename, combined)


def main():
    models = ['nov13.pt', 'LLOnly180ep.pt', 'LLOnly120.pt', 'april9120sLLO.pt']
    results = {}

    output_dir = 'output_frames'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model in models:
        ious, dices, accuracies, speeds = [], [], [], []
        data_generator = load_data()

        total_images = len(sorted(os.listdir('Drivable-area-model-8/valid/images')))

        lane_model = YOLO(model, task='segment')

        pbar = tqdm(data_generator, total=total_images)
        frame_number = 0
        for image, label in pbar:
            start_time = time.time()
            prediction = lane_model.predict(image, conf=0.5, verbose=False)[0].masks
            if prediction is not None:
                prediction = prediction.xy[0]
                width, height = image.size
                prediction = polygon_to_mask(prediction, (height, width))
            else:
                prediction = np.zeros((image.height, image.width))
            end_time = time.time()

            ground_truth = np.array(label)
            prediction = np.array(prediction)

            iou_score, dice_coefficient, pixel_accuracy = compute_metrics(ground_truth, prediction)

            ious.append(iou_score)
            dices.append(dice_coefficient)
            accuracies.append(pixel_accuracy)
            speeds.append(end_time - start_time)

            visualize_and_save(image, ground_truth, prediction, frame_number, output_dir)
            frame_number += 1
            
            pbar.set_description(f"Model: {model}, IoU: {np.mean(ious):.4f}, Dice: {np.mean(dices):.4f}, Accuracy: {np.mean(accuracies):.4f}, Speed: {1 / np.mean(speeds):.2f} FPS")

        results[model] = {
            'iou': np.mean(ious),
            'dice': np.mean(dices),
            'accuracy': np.mean(accuracies),
            'speed': np.mean(speeds),
            'total_time': np.sum(speeds),
            'fps': 1 / np.mean(speeds),
            'score': (np.mean(ious) + np.mean(dices) + np.mean(accuracies) + 1 / np.mean(speeds)) / 4
        }

        os.system(f"ffmpeg -y -framerate 5 -i {output_dir}/frame_%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {model}.mp4")

        # Remove the frames after creating the video
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    best_model = max(results, key=lambda x: results[x]['score'])
    best_iou = max(results, key=lambda x: results[x]['iou'])
    best_dice = max(results, key=lambda x: results[x]['dice'])
    best_accuracy = max(results, key=lambda x: results[x]['accuracy'])
    fastest_model = min(results, key=lambda x: results[x]['speed'])

    with open('results.json', 'w') as f:
        json.dump(results, f)

    print(f"Best model: {best_model}")
    print(f"Best IoU model: {best_iou}")
    print(f"Best Dice model: {best_dice}")
    print(f"Best Accuracy model: {best_accuracy}")
    print(f"Fastest model: {fastest_model}")


if __name__ == '__main__':
    main()