import os
import itertools
import skimage.io
import skimage.color
import torch
import pathlib
import tqdm
import random 
import cv2

COLAB = False

lilypond_path = r'C:\Program Files (x86)\LilyPond\usr\bin\lilypond.exe'
data_root_path = r'C:\Users\simon\Coding\ML\aipollo\rnntorch\data\ '[:-1] if not COLAB else r'drive/My Drive/Colab Notebooks/data/'

symbols_with_duration = [x[0] + x[1] for x in list(itertools.product(['c\'', 'd\'', 'e\'', 'f\'' , 'g\'', 'a\'', 'b\'', 'r'], ['1', '2', '4', '8']))]
symbols_without_duration = [r'\bar "|"']
all_symbols = symbols_without_duration + symbols_with_duration
#num_symbols = len(symbols_with_duration) + len(symbols_without_duration) + 2

def symbol_to_number(symbol):
    if symbol == 'START':
        return 0
    elif symbol == 'STOP':
        return 1
    elif symbol in symbols_without_duration:
        return symbols_without_duration.index(symbol) + 2
    elif symbol in symbols_with_duration:
        return len(symbols_without_duration) + 2 + symbols_with_duration.index(symbol)
    else:
        raise Exception

def generate_scores(num_symbols, length_range=(23,28), num_scores=1024, start_number=0):
    data_folder = pathlib.Path(data_root_path + f'{str(num_symbols)}_{str(length_range[1])}')
    data_folder.mkdir(parents=True, exist_ok=True)

    for i in tqdm.tqdm(range(num_scores)):
        number_of_symbols = random.randint(length_range[0] - 1, length_range[1] - 1)
        symbols = []
        no_barline_next = True
        for _ in range(number_of_symbols):
            if random.random() < 0.5 and not no_barline_next:
                symbols.append(random.choice(symbols_without_duration))
                no_barline_next = True
            else:
                symbols.append(random.choice(symbols_with_duration[:num_symbols-2-len(symbols_without_duration)]))
                no_barline_next = False
        
        # Write symbols to file. These will be the ground truth labels.
        symbols_file = data_folder / (str(i + start_number) + '.txt')
        with open(symbols_file, 'w+', encoding='utf-8') as f:
            f.write(';'.join(symbols) + ';STOP')
            
        lilypond_source = r'''
        \absolute {
        \set Score.timing = ##f
        \clef treble ''' + ' '.join(symbols) + ' }'
        print(lilypond_source)

        # Write Lilypond source file and translate it.
        lilypond_source_file = data_folder / (str(i + start_number) + '.ly')
        with open(lilypond_source_file, 'w+', encoding='utf-8') as f:
            f.write(lilypond_source)

        os.system(r'"' + lilypond_path + r'"' + ' --format=png --output=' + str(data_folder) + ' ' + str(data_folder / lilypond_source_file))




class ScoreSnippetsDataset(torch.utils.data.Dataset):
    def __init__(self, image_height, image_width, num_different_symbols, max_seq_length, num_instances=None, pad_to_length=None, transform=None):
        self._max_seq_length = max_seq_length
        self._data_folder = pathlib.Path(data_root_path + f'{str(num_different_symbols)}_{str(max_seq_length)}')
        self._image_height = image_height
        self._image_width = image_width

        if not num_instances:
            self._num_instances = max(int(file_name.split('.')[0]) for file_name in os.listdir(self._data_folder) if file_name.split('.')[0].isnumeric()) + 1
        else:
            self._num_instances = num_instances
    
        self._all_train_labels = []
        self._all_train_labels_shifted = []
        for i in range(self._num_instances):
            with open(self._data_folder / (str(i) + '.txt'), 'r', encoding='utf-8') as f:
                symbols = f.read().split(';')
                symbols_as_numbers = [symbol_to_number(symbol) for symbol in symbols]
                symbols_as_numbers_shifted = [symbol_to_number('START')] + symbols_as_numbers[:-1]

                if pad_to_length:
                    if len(symbols_as_numbers) > pad_to_length:
                        raise ValueError

                    symbols_as_numbers.extend([symbol_to_number('STOP')] * (pad_to_length - len(symbols_as_numbers)))
                    symbols_as_numbers_shifted.extend([symbol_to_number('STOP')] * (pad_to_length - len(symbols_as_numbers_shifted)))
                    assert len(symbols_as_numbers) == pad_to_length

                symbols_as_numbers = torch.tensor(symbols_as_numbers)
                symbols_as_numbers_shifted = torch.tensor(symbols_as_numbers_shifted)

                self._all_train_labels.append(symbols_as_numbers)
                self._all_train_labels_shifted.append(symbols_as_numbers_shifted)

        self._all_images = []
        for i in range(self._num_instances):
            with open(self._data_folder / (str(i) + '.txt'), 'r', encoding='utf-8') as f:
                image_path = str(self._data_folder / (str(i) + '.png'))
                image = skimage.io.imread(image_path)
                image = skimage.color.rgb2gray(image)
                image = image[:self._image_height, 50:50+self._image_width]

                #DEBUG
                #cv2.imshow('foo', image)
                #cv2.waitKey(0)

                image = image.reshape((1, self._image_height, self._image_width))
                #image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
                #image /= 255.0

                image = torch.tensor(image, dtype=torch.float)
                self._all_images.append(image)   
        
        self._transform = None if not transform else transform


    def __len__(self):
        return self._num_instances

    def __getitem__(self, i):
        if torch.is_tensor(i):
            idx = i.tolist()

        '''
        image_path = str(self._data_folder / (str(i) + '.png'))
        image = skimage.io.imread(image_path)
        image = skimage.color.rgb2gray(image)
        image = image[:self._image_height, 50:50+self._image_width]

        #DEBUG
        #cv2.imshow('foo', image)
        #cv2.waitKey(0)

        image = image.reshape((1, self._image_height, self._image_width))
        #image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
        #image /= 255.0

        image = torch.tensor(image, dtype=torch.double)
        image = image.type(torch.FloatTensor)
        '''

        images = self._all_images[i]
        if self._transform:
            images = self._transform(images)

        return (images, self._all_train_labels[i], self._all_train_labels_shifted[i])