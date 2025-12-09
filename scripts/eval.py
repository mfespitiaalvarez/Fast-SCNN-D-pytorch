import os
import sys
import torch
import torch.utils.data as data

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from models.fast_scnn_d import get_fast_scnn as get_fast_scnn_d
from utils.metric import SegmentationMetric
from utils.visualize import get_color_pallete

from train import parse_args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        # output folder
        self.outdir = 'test_result'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)
        # create network
        # Use fast_scnn_d model if dataset is citys_d or model is fast_scnn_d
        if args.model == 'fast_scnn_d' or args.dataset == 'citys_d':
            self.model = get_fast_scnn_d(dataset=args.dataset, aux=args.aux, pretrained=False).to(args.device)
        else:
            self.model = get_fast_scnn(args.dataset, aux=args.aux, pretrained=False).to(args.device)
        
        # Load checkpoint if provided
        if args.resume:
            if os.path.isfile(args.resume):
                print('Loading checkpoint from {}...'.format(args.resume))
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
                # Handle DataParallel models (remove 'module.' prefix if present)
                if any(key.startswith('module.') for key in checkpoint.keys()):
                    # Model was saved with DataParallel, create new state dict without 'module.'
                    new_checkpoint = {}
                    for k, v in checkpoint.items():
                        new_checkpoint[k.replace('module.', '')] = v
                    checkpoint = new_checkpoint
                self.model.load_state_dict(checkpoint)
                print('Finished loading checkpoint!')
            else:
                raise FileNotFoundError('Checkpoint file not found: {}'.format(args.resume))
        else:
            print('Warning: No checkpoint specified. Using randomly initialized model.')

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        self.model.eval()
        for i, (image, label) in enumerate(self.val_loader):
            image = image.to(self.args.device)

            outputs = self.model(image)

            pred = torch.argmax(outputs[0], 1)
            pred = pred.cpu().data.numpy()
            label = label.numpy()

            self.metric.update(pred, label)
            pixAcc, mIoU = self.metric.get()
            print('Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' % (i + 1, pixAcc * 100, mIoU * 100))

            predict = pred.squeeze(0)
            mask = get_color_pallete(predict, self.args.dataset)
            mask.save(os.path.join(self.outdir, 'seg_{}.png'.format(i)))


if __name__ == '__main__':
    args = parse_args()
    evaluator = Evaluator(args)
    print('Testing model: ', args.model)
    print('Testing dataset: ', args.dataset)
    if args.resume:
        print('Checkpoint: ', args.resume)
    evaluator.eval()
    # Print final metrics
    pixAcc, mIoU = evaluator.metric.get()
    print('\n' + '='*50)
    print('Final Results:')
    print('Pixel Accuracy: {:.3f}%'.format(pixAcc * 100))
    print('Mean IoU: {:.3f}%'.format(mIoU * 100))
    print('='*50)
