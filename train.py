from dataset import processes
import os 
import torch  
import argparse
import logging
import yaml
import wandb  
import numpy as np 
import time 
import random 
import math 
from copy import deepcopy
from threading import Thread
from torch import nn 
from torch import optim
import test  

from torch.cuda import amp
import torch.distributed as dist
from torch.optim import lr_scheduler
from pathlib import Path 
from tqdm import tqdm 

from dataset.dataset import build_dataloader
from models.model import Model
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.torch_utils import intersect_dicts, ModelEMA, is_parallel
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from evaltool.valid import Validator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def check_model(config):
    model = Model(config['model'])
    temp_tensor = torch.zeros(1, 3, 640, 640)
    res = model(temp_tensor)
    print(res)

# def check_dataset(config):
#     dataset = ImageDataset()    

class Trainer():
    def __init__(self, hyp, opt):
        self.hyp = hyp 
        self.opt = opt 
        self.model = None 
        self.validor = Validator() 
        
    
    def freeze(self, model):
        # Freeze
        freeze = []  # parameter names to freeze (full or partial)
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False


    def build_model(self, weights):
        ckpt = None
        pretrained = weights.endswith('.pt')
        if pretrained:
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            model = Model(self.opt.cfg or ckpt['model'].yaml, ch=3, device=device).to(device)  # create
            exclude = ['anchor'] if self.opt.cfg or self.hyp.get('anchors') else []  # exclude keys
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(state_dict, strict=False)  # load
            logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
        else:
            model = Model(self.opt.cfg, ch=3, device=device).to(device)  # create
        return pretrained, ckpt, model 


    def build_optim(self, total_batch_size):
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
        logger.info(f"Scaled weight_decay = {self.hyp['weight_decay']}")

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        if self.opt.adam:
            optimizer = optim.Adam(pg0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        optimizer.add_param_group({'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2
        return optimizer

    def build_scheduler(self, epochs):
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        if opt.linear_lr:
            lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear
        else:
            lf = one_cycle(1, self.hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        return scheduler, lf 


    def train(self, hyp, opt, device):
        
        opt = self.opt
        hyp = self.hyp 
        nbs = 64  # nominal batch size
        logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        save_dir, epochs, batch_size, weights = \
            Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights
        total_batch_size = batch_size 
        plots = True  # create plots
        # Directories
        wdir = save_dir / 'weights'
        wdir.mkdir(parents=True, exist_ok=True)  # make dir
        last = wdir / 'last.pt'
        best = wdir / 'best.pt'
        results_file = save_dir / 'results.txt'

        # Save run settings
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(hyp, f, sort_keys=False)
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.dump(vars(opt), f, sort_keys=False)

        # Configure
        
        cuda = device.type != 'cpu'
        init_seeds(2)
        with open(opt.data) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        # with torch_distributed_zero_first(rank):
        #     check_dataset(data_dict)  # check

        train_path = data_dict['train']
        test_path = data_dict['val']
      #  print(data_dict)
    
        # Model
        pretrained, ckpt, self.model = self.build_model(weights)

        # Freeze
        # self.freeze(model)

        # Optimizer
        self.optimizer = self.build_optim(total_batch_size)

        self.scheduler, lf = self.build_scheduler(epochs=epochs)

        
        wandb = False 
        # Logging
        if  wandb and wandb.run is None:
            opt.hyp = hyp  # add hyperparameters
            wandb_run = wandb.init(config=opt, resume="allow",
                                project='YOLODB' if opt.project == 'runs/train' else Path(opt.project).stem,
                                name=save_dir.stem,
                                entity=opt.entity,
                                id=ckpt.get('wandb_id') if ckpt is not None else None)
        loggers = {'wandb': wandb}  # loggers dict

        # EMA
       # ema = ModelEMA(self.model) if rank in [-1, 0] else None
        ema = ModelEMA(self.model) 

        # Resume
        start_epoch, best_fitness = 0, 0.0
        if pretrained:
            # Optimizer
            if ckpt and ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # EMA
            if ema and ckpt.get('ema'):
                ema.ema.load_state_dict(ckpt['ema'][0].float().state_dict())
                ema.updates = ckpt['ema'][1]

            # Results
            if ckpt.get('training_results') is not None:
                results_file.write_text(ckpt['training_results'])  # write results.txt

            # Epochs
            start_epoch = ckpt['epoch'] + 1
            if opt.resume:
                assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
            if epochs < start_epoch:
                logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                            (weights, ckpt['epoch'], epochs))
                epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt

 

        imgsz, imgsz_test = opt.img_size

        
        train_process = data_dict['process']['train']
        val_process = data_dict['process']['val']
        # Trainloader
        dataloader, dataset = build_dataloader(train_path, imgsz, batch_size, opt,
                                                hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, 
                                                workers=opt.workers,
                                                image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '),
                                                process_list=train_process, mode='train')
    
        num_of_batches = len(dataloader)  # number of batches
       
    
        testloader = build_dataloader(test_path, imgsz_test, batch_size, opt,  # testloader
                                    hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                    workers=opt.workers,
                                    pad=0.5, prefix=colorstr('val: '), process_list=val_process, mode='valid')[0]

        # if not opt.resume:
        #     labels = np.concatenate(dataset.labels, 0)
        #    # c = torch.tensor(labels[:, 0])  # classes
        #     print(labels)
        #     if plots:
        #         plot_labels(labels, save_dir, loggers)
                # if tb_writer:
                #     tb_writer.add_histogram('classes', c, 0)

            

        # Start training
        t0 = time.time()
        nw = max(round(hyp['warmup_epochs'] * num_of_batches), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        nc = 1
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        self.scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = amp.GradScaler(enabled=False)
        logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                    f'Using {dataloader.num_workers} dataloader workers\n'
                    f'Logging results to {save_dir}\n'
                    f'Starting training for {epochs} epochs...')

        for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
            self.model.train()

            mloss = torch.zeros(4, device=device)  # mean losses
        
            pbar = enumerate(dataloader)
            logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
           
            pbar = tqdm(pbar, total=num_of_batches)  # progress bar
            self.optimizer.zero_grad()
            for i, batch in pbar:  # batch -------------------------------------------------------------
                ni = i + num_of_batches * epoch  # number integrated batches (since train start)
                # if i> 1:
                #     break 
                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # Multi-scale
                # if opt.multi_scale:
                #     sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                #     sf = sz / max(imgs.shape[2:])  # scale factor
                #     if sf != 1:
                #         ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                #         imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=False):
                    loss, pred, metrics = self.model.compute_loss(batch, training=True)
                    
               # print(loss)
                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni % accumulate == 0:
                    scaler.step(self.optimizer)  # optimizer.step
                    scaler.update()
                    self.optimizer.zero_grad()
                    if ema:
                        ema.update(self.model)

                if isinstance(loss, dict):
                    line = []
                    loss = torch.tensor(0.).cuda()
                    for key, l_val in loss.items():
                        loss += l_val.mean()
                        line.append('loss_{0}:{1:.4f}'.format(key, l_val.mean()))
                else:
                    loss = loss.mean()
                # Print
              
                # for name, metric in metrics.items():
                #     print('%s: %6f' % (name, metric.mean()))


                mloss = (mloss * i + loss) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, 1, batch['image'].shape[-1])
                pbar.set_description(s)

                # Plot
                # if plots and ni < 3:
                #     f = save_dir / f'train_batch{ni}.jpg'  # filename
                #     Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                #     # if tb_writer:
                #     #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                #     #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
                # elif plots and ni == 10 and wandb:
                #     wandb.log({"Mosaics": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob('train*.jpg')
                #                         if x.exists()]}, commit=False)

                # end batch ------------------------------------------------------------------------------------------------
            # end epoch ----------------------------------------------------------------------------------------------------

            # Scheduler
            lr = [x['lr'] for x in self.optimizer.param_groups]  # for tensorboard
            self.scheduler.step()

           
            #ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs

            if not opt.notest or final_epoch:  # Calculate mAP
                valid_result = self.validor.validate({'test':testloader}, self.model, epoch, num_of_batches*epoch)
            print(valid_result)
            # Write
            # with open(results_file, 'a') as f:
            #     f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            # if len(opt.name) and opt.bucket:
            #     os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            # tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
            #         'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
            #         'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
            #         'x/lr0', 'x/lr1', 'x/lr2']  # params
            # for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
            #     # if tb_writer:
            #     #     tb_writer.add_scalar(tag, x, epoch)  # tensorboard
            #     if wandb:
            #         wandb.log({tag: x}, step=epoch, commit=tag == tags[-1])  # W&B

            # Update best mAP
            #fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            
            # if fi > best_fitness:
            #     best_fitness = fi

            # # Save model
            # if (not opt.nosave) or (final_epoch):  # if save
            #     ckpt = {'epoch': epoch,
            #             'best_fitness': best_fitness,
            #          #   'training_results': results_file.read_text(),
            #             'model': ema.ema if final_epoch else deepcopy(
            #                 self.model.module if is_parallel(self.model) else self.model).half(),
            #             'ema': (deepcopy(ema.ema).half(), ema.updates),
            #             'optimizer': self.optimizer.state_dict(),
            #             'wandb_id': wandb_run.id if wandb else None}

            #     # Save last, best and delete
            #     torch.save(ckpt, last)
            #     if best_fitness == fi:
            #         torch.save(ckpt, best)
            #     del ckpt

            # end epoch ----------------------------------------------------------------------------------------------------
        # end training

        #if rank in [-1, 0]:
        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)
     
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb.log({"Results": [wandb.Image(str(save_dir / f), caption=f) for f in files
                                    if (save_dir / f).exists()]})
                if opt.log_artifacts:
                    wandb.log_artifact(artifact_or_path=str(final), type='model', name=save_dir.stem)

        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        # if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
        #     for m in (last, best) if best.exists() else (last):  # speed, mAP tests
        #         results, _, _ = test.test(opt.data,
        #                                 batch_size=batch_size * 2,
        #                                 imgsz=imgsz_test,
        #                                 conf_thres=0.001,
        #                                 iou_thres=0.7,
        #                                 model=attempt_load(m, device).half(),
        #                                 single_cls=opt.single_cls,
        #                                 dataloader=testloader,
        #                                 save_dir=save_dir,
        #                                 save_json=True,
        #                                 plots=False)

     

        wandb.run.finish() if wandb and wandb.run else None
        torch.cuda.empty_cache()
        return results



     


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
   # parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
   # parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
   # parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
   # parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
   # parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
   # parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--visualize', action='store_true', help='visualize')

    
    opt = parser.parse_args()
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
   # print(hyp)
     

         # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok )  # increment run
       # print(opt.cfg)
    #check_model(hyp)

    device = torch.device('cuda:2')

    trainer = Trainer(hyp, opt)
    trainer.train(hyp, opt, device)