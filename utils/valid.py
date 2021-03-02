    
import logging
import os 
from tqdm import tqdm 
from .representer import SegDetectorRepresenter
from .measurer import ICDARDetectionMeasurer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
class Validator():
    def __init__(self):
        self.representer = SegDetectorRepresenter()
        self.measurer = ICDARDetectionMeasurer()


    def validate(self, validation_loaders, model, epoch, step, visualize=False):
        all_matircs = {}
        model.eval()
        for name, loader in validation_loaders.items():
            if visualize:
                metrics, vis_images = self.validate_step(loader, model, True)
                logger.images(
                    os.path.join('vis', name), vis_images, step)
            else:
                metrics, vis_images = self.validate_step(loader, model, False)
            for _key, metric in metrics.items():
                key = name + '/' + _key
                if key in all_matircs:
                    all_matircs[key].update(metric.val, metric.count)
                else:
                    all_matircs[key] = metric

        for key, metric in all_matircs.items():
            logger.info('%s : %f (%d)' % (key, metric.avg, metric.count))
        self.logger.metrics(epoch, self.steps, all_matircs)
        model.train()
        return all_matircs

    def validate_step(self, data_loader, model, visualize=False):
        raw_metrics = []
        vis_images = dict()
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            pred = model.compute_loss(batch, training=False)
            output = self.representer.represent(batch, pred)
            raw_metric, interested = self.measurer.validate_measure(batch, output)
            raw_metrics.append(raw_metric)

            if visualize and self.visualizer:
                vis_image = self.visualizer.visualize(batch, output, interested)
                vis_images.update(vis_image)
        metrics = self.measurer.gather_measure(raw_metrics, self.logger)
        return metrics, vis_images