import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import os
import numpy as np
from model.deeplab import Res_Deeplab


def generate_pseudo_label(model, save_path, target_loader):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.eval()
    model.cuda()

    predicted_label = np.zeros((len(target_loader), 512, 1024))
    predicted_prob = np.zeros((len(target_loader), 512, 1024))
    image_name = []

    interp = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)

    for index, batch in enumerate(target_loader):
        if index % 100 == 0:
            print('%d processd' % index)
        image, _, name = batch
        output = model(Variable(image).cuda())
        output = nn.functional.softmax(output, dim=1)
        output = interp(output).cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)

        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        predicted_label[index] = label.copy()
        predicted_prob[index] = prob.copy()
        image_name.append(name[0])
        
    thres = []
    for i in range(19):
        x = predicted_prob[predicted_label==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x)*0.5))])
    print(thres)
    thres = np.array(thres)
    thres[thres>0.9]=0.9
    print(thres)
    for index in range(len(target_loader)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(19):
            label[(prob<thres[i])*(label==i)] = 255  
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split('/')[-1]
        output.save('%s/%s' % (save_path, name))
