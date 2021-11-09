import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from paz.datasets.utils import get_class_names
from paz.abstract import ProcessingSequence
from pipelines import ProcessGrayImage
from paz.datasets import FER, FERPlus
from paz.models.classification import MiniXception

description = 'Emotion recognition training'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-d', '--dataset', default='FER', type=str,
                    choices=['FERPlus', 'FER'])
parser.add_argument('-p', '--data_path', type=str,
                    default='/media/deepan/externaldrive1/project_repos/'
                            'paz_versions/fer/fer2013/',
                    help='Default root data path')
parser.add_argument('-t', '--evaluation_splits', nargs='+', type=str,
                    default=['test'], help='Splits used for evaluation')
parser.add_argument('-v', '--validation_split', default='test', type=str,
                    help='Split used for validation')
parser.add_argument('-m', '--model', default='MINI-XCEPTION', type=str,
                    choices=['MINI-XCEPTION'])
args = parser.parse_args()
size = 48
batch_size = 1
dataset = args.dataset
data_path = os.path.join(args.data_path, dataset)
num_classes = 7 if dataset=='FER' else 8

pipeline = ProcessGrayImage(size=size, num_classes=num_classes)
name_to_manager = {'FER': FER, 'FERPlus': FERPlus}
data_manager = name_to_manager[dataset](path=data_path, split='train')
data = data_manager.load_data()
sequence = ProcessingSequence(pipeline, batch_size, data)

model = MiniXception((size, size, 1), num_classes, weights=dataset)

for n, i in enumerate(sequence):
    print("######")
    image = i[0]['image']
    print(np.max(image), np.min(image))
    label = i[1]['label']
    plt.imsave('im' + str(n)+'.jpg', image[0][..., 0])
    x = model(image)
    prob = np.amax(x)
    pred = np.argmax(x)
    true = np.argmax(label[0])
    class_names = get_class_names('FER')
    print('Results on image number: ', n)
    print('Probability: ', prob)
    print('Predicted class: ', class_names[pred])
    print('True class: ', class_names[true])
