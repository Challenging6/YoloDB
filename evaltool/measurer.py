import os
import subprocess
import shutil

import numpy as np
import json

import numpy as np

from .icdar2015_eval.detection.iou import DetectionIoUEvaluator



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self



class QuadMeasurer():
    def __init__(self, **kwargs):
        self.evaluator = DetectionIoUEvaluator()

    def measure(self, batch, output, is_output_polygon=False, box_thresh=0.6):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        output: (polygons, ...)
        '''
        results = []
        gt_polyons_batch = batch['polygons']
        ignore_tags_batch = batch['ignore_tags']
        pred_polygons_batch = np.array(output[0])
        pred_scores_batch = np.array(output[1])
        for polygons, pred_polygons, pred_scores, ignore_tags in\
                zip(gt_polyons_batch, pred_polygons_batch, pred_scores_batch, ignore_tags_batch):
            gt = [dict(points=polygons[i], ignore=ignore_tags[i])
                  for i in range(len(polygons))]
            if is_output_polygon:
                pred = [dict(points=pred_polygons[i])
                        for i in range(len(pred_polygons))]
            else:
                pred = []
                # print(pred_polygons.shape)
                for i in range(pred_polygons.shape[0]):
                    if pred_scores[i] >= box_thresh:
                        # print(pred_polygons[i,:,:].tolist())
                        pred.append(dict(points=pred_polygons[i,:,:].tolist()))
                # pred = [dict(points=pred_polygons[i,:,:].tolist()) if pred_scores[i] >= box_thresh for i in range(pred_polygons.shape[0])]
            results.append(self.evaluator.evaluate_image(gt, pred))
        return results

    def validate_measure(self, batch, output, is_output_polygon=False, box_thresh=0.6):
        return self.measure(batch, output, is_output_polygon, box_thresh)

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output),\
            np.linspace(0, batch['image'].shape[0]).tolist()

    def gather_measure(self, raw_metrics):
        raw_metrics = [image_metrics
                       for batch_metrics in raw_metrics
                       for image_metrics in batch_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val /\
            (precision.val + recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }


class ICDARDetectionMeasurer():
    def __init__(self, **kwargs):
        self.visualized = False

    def measure(self, batch, output):
        pairs = []
        for i in range(len(batch[-1])):
            pairs.append((batch[-1][i], output[i][0]))
        return pairs

    def validate_measure(self, batch, output):
        return self.measure(batch, output), [int(self.visualized)]

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch[0].shape[0]).tolist()

    def gather_measure(self, name, raw_metrics, logger):
        save_dir = os.path.join(logger.log_dir, name)
        shutil.rmtree(save_dir, ignore_errors=True)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        log_file_path = os.path.join(save_dir, name + '.log')
        count = 0
        for batch_pairs in raw_metrics:
            for _filename, boxes in batch_pairs:
                boxes = np.array(boxes).reshape(-1, 8).astype(np.int32)
                filename = 'res_' + _filename.replace('.jpg', '.txt')
                with open(os.path.join(save_dir, filename), 'wt') as f:
                    if len(boxes) == 0:
                        f.write('')
                    for box in boxes:
                        f.write(','.join(map(str, box)) + '\n')
                count += 1

        self.packing(save_dir)
        try:
            raw_out = subprocess.check_output(['python assets/ic15_eval/script.py -m=' + name
                                               + ' -g=assets/ic15_eval/gt.zip -s=' +
                                               os.path.join(save_dir, 'submit.zip') +
                                               '|tee -a ' + log_file_path],
                                              timeout=30, shell=True)
        except subprocess.TimeoutExpired:
            return {}
        raw_out = raw_out.decode().replace('Calculated!', '')
        dict_out = json.loads(raw_out)
        return {k: AverageMeter().update(v, n=count) for k, v in dict_out.items()}

    def packing(self, save_dir):
        pack_name = 'submit.zip'
        os.system(
            'zip -r -j -q ' +
            os.path.join(save_dir, pack_name) + ' ' + save_dir + '/*.txt')
