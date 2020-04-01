import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
from eval_and_compute.evaluate import eval
from eval_and_compute.compute import compute_mIoU

from model.deeplab import Res_Deeplab
from model.deeplab_vgg import DeeplabVGG
from utils.loss import CrossEntropy2d
from model.discriminator import FCDiscriminator

from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.cityscapes_dataset_label import cityscapesDataSetLabel

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
IGNORE_LABEL = 255
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
POWER = 0.9
RANDOM_SEED = 1234
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
WEIGHT_DECAY = 0.0005
LEARNING_RATE_D = 1e-4
LAMBDA_ADV = 0.001

MODEL = 'ResNet'
SOURCE ='GTA5'
TARGET = 'cityscapes'
STAGE = 1
SET = 'train'


EXPERIMENT_NAME = 'GTA5_TO_CITYSCAPES'

RESTORE_FROM = 'Path to initial weight'
SNAPSHOT_DIR = 'Path to snapshot/' + EXPERIMENT_NAME + '/'
CITYSCAPES_EVAL_DIR = 'Path to evaluation' + EXPERIMENT_NAME + '/'
OUTPUT_FILE = 'Path to mIoU output' + EXPERIMENT_NAME + '.txt'
LOG_DIR = './log/' + EXPERIMENT_NAME
CHECKPOINT = ''

INPUT_SIZE_SOURCE = '1024,512'
TRANSLATED_DATA_DIRECTORY = 'Path to translated dataset'
STYLIZED_DATA_DIRECTORY = 'Path to stylized dataset'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'

DATA_DIRECTORY_TARGET = 'Path to cityscapes'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed farguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--translated-data-dir", type=str, default=TRANSLATED_DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--stylized-data-dir", type=str, default=STYLIZED_DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size-source", type=str, default=INPUT_SIZE_SOURCE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--source", type=str, default=SOURCE)
    parser.add_argument("--output-file", type=str, default=OUTPUT_FILE)
    parser.add_argument("--cityscapes-eval-dir", type=str, default=CITYSCAPES_EVAL_DIR)
    parser.add_argument("--lambda-adv", type=float, default=LAMBDA_ADV,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def load_checkpoint(model, model_D, optimizer, filename=''):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_iter = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['model'])
        model_D.load_state_dict(checkpoint['model_D'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (iter {})".format(filename, checkpoint['iter']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, model_D, optimizer, start_iter


def main():
    """Create the model and start the training."""
    w, h = map(int, args.input_size_source.split(','))
    input_size_source = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    # Create network
    if args.model == 'ResNet':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http':
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)

        # model.load_state_dict(saved_state_dict)

    elif args.model == 'VGG':
        model = DeeplabVGG(num_classes=args.num_classes, pretrained=True, vgg16_caffe_path=args.restore_from)

        # saved_state_dict = torch.load(args.restore_from)
        # model.load_state_dict(saved_state_dict)

    optimizer = optim.SGD(model.optim_parameters(args),lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    model.train()
    model.cuda(args.gpu)
    cudnn.benchmark = True

    #Discrimintator setting
    model_D = FCDiscriminator(num_classes=args.num_classes)
    model_D.train()
    model_D.cuda(args.gpu)

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    # labels for adversarial training
    source_adv_label = 0
    target_adv_label = 1

    #Dataloader
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        GTA5DataSet(args.translated_data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size_source,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)


    style_trainloader = data.DataLoader(
        GTA5DataSet(args.stylized_data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size_source,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    style_trainloader_iter = enumerate(style_trainloader)


    if STAGE == 1:
        targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                              max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                              crop_size=input_size_target,
                                                              mean=IMG_MEAN,
                                                              set=args.set),
                                       batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                       pin_memory=True)

        targetloader_iter = enumerate(targetloader)

    else:
        #Dataloader for self-training
        targetloader = data.DataLoader(cityscapesDataSetLabel(args.data_dir_target, args.data_list_target,
                                                         max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                         crop_size=input_size_target,
                                                         mean=IMG_MEAN,
                                                         set=args.set, label_folder='Path to generated pseudo labels'),
                                       batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                       pin_memory=True)

        targetloader_iter = enumerate(targetloader)

    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)


    # load checkpoint
    model, model_D, optimizer, start_iter = load_checkpoint(model, model_D, optimizer, filename=args.snapshot_dir + 'checkpoint_' + CHECKPOINT + '.pth.tar')

    for i_iter in range(start_iter, args.num_steps):
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)


        #train segementation network
        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # train with source
        if STAGE == 1:
            if i_iter % 2 == 0:
                _, batch = next(trainloader_iter)
            else:
                _, batch = next(style_trainloader_iter)

        else:
            _, batch = next(trainloader_iter)


        image_source, label, _, _ = batch
        image_source = Variable(image_source).cuda(args.gpu)

        pred_source = model(image_source)
        pred_source = interp(pred_source)

        loss_seg_source = loss_calc(pred_source, label, args.gpu)
        loss_seg_source_value = loss_seg_source.item()
        loss_seg_source.backward()

        if STAGE == 2:
            # train with target
            _, batch = next(targetloader_iter)
            image_target, target_label, _, _ = batch
            image_target = Variable(image_target).cuda(args.gpu)

            pred_target = model(image_target)
            pred_target = interp_target(pred_target)

            #target segmentation loss
            loss_seg_target = loss_calc(pred_target, target_label, gpu = args.gpu)
            loss_seg_target.backward()

        # optimize
        optimizer.step()


        if STAGE == 1:
            #output-level adversarial training
            D_output_target = model_D(F.softmax(pred_target))
            loss_adv = bce_loss(D_output_target, Variable(torch.FloatTensor(D_output_target.data.size()).fill_(source_adv_label)).cuda(args.gpu))
            loss_adv = loss_adv * args.lambda_adv
            loss_adv.backward()

            #train discriminator
            for param in model_D.parameters():
                param.requires_grad = True

            pred_source = pred_source.detach()
            pred_target = pred_target.detach()

            D_output_source = model_D(F.softmax(pred_source))
            D_output_target = model_D(F.softmax(pred_target))

            loss_D_source = bce_loss(D_output_source, Variable(torch.FloatTensor(D_output_source.data.size()).fill_(source_adv_label)).cuda(args.gpu))
            loss_D_target = bce_loss(D_output_target, Variable(torch.FloatTensor(D_output_target.data.size()).fill_(target_adv_label)).cuda(args.gpu))

            loss_D_source = loss_D_source / 2
            loss_D_target = loss_D_target / 2

            loss_D_source.backward()
            loss_D_target.backward()

            #optimize
            optimizer_D.step()


        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg_source = {2:.5f}'
            .format(
            i_iter, args.num_steps, loss_seg_source_value
        ))


        if i_iter % args.save_pred_every == 0:
            print ('taking snapshot ...')
            state = {'iter' : i_iter, 'model' : model.state_dict(), 'model_D' : model_D.state_dict(), 'optimizer' : optimizer.state_dict()}
            torch.save(state, osp.join(args.snapshot_dir, 'checkpoint_' + str(i_iter) + '.pth.tar'))
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_D_' + str(i_iter) + '.pth'))

            cityscapes_eval_dir = osp.join(args.cityscapes_eval_dir, str(i_iter))
            if not os.path.exists(cityscapes_eval_dir):
                os.makedirs(cityscapes_eval_dir)

            eval(osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'), cityscapes_eval_dir, i_iter)

            iou19, iou13, iou = compute_mIoU(cityscapes_eval_dir, i_iter)
            outputfile = open(args.output_file, 'a')
            outputfile.write(str(i_iter) + '\t' + str(iou19) + '\t' + str(iou.replace('\n',' ')) + '\n')
            outputfile.close()

if __name__ == '__main__':
    main()
