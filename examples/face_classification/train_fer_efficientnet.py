import os
import json
import argparse
from datetime import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model

from paz.abstract import ProcessingSequence, SequentialProcessor
from paz.datasets import FER, FERPlus
import paz.processors as pr


class ProcessGrayImage(SequentialProcessor):
    def __init__(self, size, num_classes, generator=None):
        super(ProcessGrayImage, self).__init__()
        self.size = size
        self.process = SequentialProcessor([pr.ExpandDims(-1)])
        if generator is not None:
            self.process.add(pr.ImageDataProcessor(generator))
        self.process.add(pr.ResizeImage((size, size)))
        self.process.add(pr.CastImage(float))
        self.process.add((pr.ExpandDims(-1)))
        self.process.add(pr.Stacker(-1))
        self.add(pr.UnpackDictionary(['image', 'label']))
        self.add(pr.ExpandDomain(self.process))
        self.add(pr.SequenceWrapper({0: {'input_1': [size, size, 3]}},
                                    {1: {'dense_1': [num_classes]}}))

description = 'Emotion recognition training'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-s', '--save_path', default='experiments', type=str,
                    help='Path for saving evaluations')
parser.add_argument('-d', '--dataset', default='FER', type=str,
                    choices=['FERPlus', 'FER'])
parser.add_argument('-m', '--model', default='EfficientNetB0', type=str,
                    choices=['EfficientNetB0'])
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    help='Batch size used during optimization')
parser.add_argument('-e', '--epochs', default=100, type=int,
                    help='Number of epochs before finishing')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Learning rate')
parser.add_argument('-o', '--stop_patience', default=5, type=int,
                    help='Early stop patience')
parser.add_argument('-u', '--reduce_patience', default=2, type=int,
                    help='Reduce learning rate patience')
parser.add_argument('-l', '--run_label', default='RUN_00', type=str,
                    help='Label used to distinguish between different runs')
parser.add_argument('-t', '--evaluation_splits', nargs='+', type=str,
                    default=['test'], help='Splits used for evaluation')
parser.add_argument('-v', '--validation_split', default='test', type=str,
                    help='Split used for validation')
parser.add_argument('--time', type=str,
                    default=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
parser.add_argument('-p', '--data_path', type=str,
                    default='/media/deepan/externaldrive1/project_repos/'
                            'paz_versions/fer/fer2013/',
                    help='Default root data path')
args = parser.parse_args()
learning_rate = args.learning_rate

splits = ['train'] + args.evaluation_splits
if args.validation_split not in splits:
    splits = splits + [args.validation_split]

print('Splits of data for loading: ', splits)

# loading data and instantiating data managers
name_to_manager = {'FER': FER, 'FERPlus': FERPlus}
data_managers, datasets = {}, {}
for split in splits:
    data_path = os.path.join(args.data_path, args.dataset)
    kwargs = {'path': data_path}
    data_manager = name_to_manager[args.dataset](split=split, **kwargs)
    data_managers[split] = data_manager
    datasets[split] = data_manager.load_data()

# data generator and augmentations
generator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True)

# instantiating sequencers
sequencers = {}
for split in splits:
    data_manager = data_managers[split]
    size, num_classes = data_manager.image_size[0], data_manager.num_classes
    if split == 'train':
        pipeline = ProcessGrayImage(size, num_classes, generator)
    else:
        pipeline = ProcessGrayImage(size, num_classes)
    sequencers[split] = ProcessingSequence(
        pipeline, args.batch_size, datasets[split])

# instantiating model
name_to_model = {'EfficientNetB0': EfficientNetB0}
base_model = name_to_model[args.model]
input_shape = pipeline.processors[-1].inputs_info[0]['input_1']
num_classes = pipeline.processors[-1].labels_info[1]['dense_1'][0]

print('Dataset stats: ', num_classes, input_shape)

base_model = base_model(input_shape=tuple(input_shape), classes=num_classes,
                        include_top=False)
for layer in base_model.layers:
    layer.trainable = True
x = Flatten()(base_model.output)
x = Dense(1000, activation='relu')(x)
out = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=out)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# creating training dataset
experiment_label = '_'.join([args.dataset, model.name, args.run_label])
experiment_path = os.path.join(args.save_path, experiment_label)
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)

# setting additional callbacks
log_filename = os.path.join(experiment_path, 'optimization.log')
log = keras.callbacks.CSVLogger(log_filename)
stop = keras.callbacks.EarlyStopping(patience=args.stop_patience)
save_filename = os.path.join(experiment_path, 'model.hdf5')
save = keras.callbacks.ModelCheckpoint(save_filename, save_best_only=True)
plateau = keras.callbacks.ReduceLROnPlateau(patience=args.reduce_patience)
callbacks = [log, stop, save, plateau]

# saving hyper-parameters and model summary
with open(os.path.join(experiment_path, 'hyperparameters.json'), 'w') as filer:
    json.dump(args.__dict__, filer, indent=4)
with open(os.path.join(experiment_path, 'model_summary.txt'), 'w') as filer:
    model.summary(print_fn=lambda x: filer.write(x + '\n'))

# starting optimization
model.fit_generator(
    sequencers['train'],
    epochs=args.epochs,
    validation_data=sequencers[args.validation_split],
    callbacks=callbacks,
    verbose=1)
