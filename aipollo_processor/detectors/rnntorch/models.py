import torch
import torchvision
import math
from torch import sub
from torchvision.transforms.transforms import CenterCrop, ColorJitter
import data 
import cv2

batch_size=32


class CRNN(torch.nn.Module):
    def __init__(self, image_height, image_width, alphabet_size, num_subimage_features, pretrained=True):
        super(CRNN, self).__init__()
        self._alphabet_size = alphabet_size
        self._pretrained=pretrained

        # Convolutional network
        if pretrained:
            self._base_model = torchvision.models.resnet18(pretrained=True)
            self._base_model.fc = torch.nn.Linear(in_features=512, out_features=num_subimage_features)
        else:
            self._conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=2) 
            self._conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=2) 
            self._conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2), stride=2) 
            self._conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=2) 
            self._fc = torch.nn.Linear(in_features=13824, out_features=num_subimage_features)

        # RNN
        self._rnn = torch.nn.LSTM(input_size=alphabet_size, hidden_size=num_subimage_features)

        # Mapping RNN hidden state to output symbol
        self._fc_hidden_to_output_symbol = torch.nn.Linear(in_features=num_subimage_features, out_features=alphabet_size)

    def forward(self, inputs):
        subimages, shifted_target = inputs
        # Pass the subimages through the CNN
        if self._pretrained:
            subimages = torch.cat([subimages, subimages, subimages], dim=1)
            subimage_features = self._base_model(subimages)
        else:
            subimage_features = torch.nn.functional.relu(self._conv4(torch.nn.functional.relu(self._conv3(torch.nn.functional.relu(self._conv2(torch.nn.functional.relu(self._conv1(subimages))))))))
            subimage_features = torch.flatten(subimage_features, start_dim=1)
            subimage_features = torch.nn.functional.relu(self._fc(subimage_features))
        subimage_features = subimage_features.view(1, batch_size, -1)

        # Compute one-hot encoding of shifted target
        shifted_target_one_hot = torch.nn.functional.one_hot(shifted_target, num_classes=self._alphabet_size)
        shifted_target_one_hot = shifted_target_one_hot.type(torch.float32) 

        # Pass the subimage features as hidden state into the RNN, 
        #output_seq, _ = self._rnn(input=shifted_target_one_hot, hx=subimage_features)
        output_seq, _ = self._rnn(shifted_target_one_hot, (subimage_features, subimage_features))
        output_seq = torch.nn.functional.relu(self._fc_hidden_to_output_symbol(output_seq))
        output_seq = torch.nn.functional.softmax(output_seq, dim=2)

        return output_seq


def train_real():
    image_height=100
    image_width=300
    max_seq_len = 5
    alphabet_size=50
    pretrained=True

    crnn = CRNN(image_height=image_height, image_width=image_width, alphabet_size=alphabet_size, num_subimage_features=256, pretrained=pretrained)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(crnn.parameters())

    aspect_ratio = max(image_width, image_height) / min(image_width, image_height)
    if pretrained:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomCrop(size=(image_height-20, image_width-20)),
            torchvision.transforms.RandomResizedCrop(size=(image_height, image_width), scale=(0.9, 0.99), ratio=(0.9*aspect_ratio, 1.1*aspect_ratio)),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize(mean=[0.43], std=[0.226]),
        ])
    else:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomCrop(size=(image_height-20, image_width-20)),
            torchvision.transforms.RandomResizedCrop(size=(image_height, image_width), scale=(0.9, 0.99), ratio=(0.9*aspect_ratio, 1.1*aspect_ratio)),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            torchvision.transforms.ToTensor(),
        ])
 
    dataset = data.ScoreSnippetsDataset(image_height=image_height, image_width=image_width, num_different_symbols=alphabet_size, max_seq_length=max_seq_len, pad_to_length=max_seq_len+1, transform=train_transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[math.floor(len(dataset)*0.9), len(dataset) - math.floor(len(dataset)*0.9)])
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0, drop_last=True)
    print(f'#instances in train / validation set: {len(train_dataset)} / {len(val_dataset)}')

    for epoch in range(1024):
        print(f'Epoch: {epoch}' )

        # Training
        crnn.train(True)
        running_loss = 0.0
        for batch in train_data_loader:
            images, labels, labels_shifted = batch
            labels = labels.permute(1, 0)
            labels_shifted = labels_shifted.permute(1, 0)
            inputs = (images, labels_shifted) # images should be is BATCH_SIZExCxHxW; shifted_labels SEQ_LENxBATCH_SIZE

            optimizer.zero_grad()
            outputs = crnn(inputs)

            # Debug
            output_seq_numbers = torch.argmax(outputs, dim=2)
            print(f'{"Prediction: ".ljust(15)} {output_seq_numbers[:, 0]}')
            print(f'{"Real: ".ljust(15)} {labels.permute(1, 0)[0]}')
            print('\n')
            cv2.imshow('Input', images[0].permute(1, 2, 0).numpy())
            cv2.waitKey(1)

            outputs = outputs.view(size=(-1, alphabet_size))
            loss = criterion(outputs, labels.reshape(shape=(-1,)))
            loss.backward()
            optimizer.step()

            #print(loss / batch_size)
            running_loss += loss

        print(f'\nAverage train loss in epoch {epoch}: {running_loss / math.floor(len(train_dataset) / batch_size):.4f}\n')
        print(f'{len(train_dataset)}')
        
        # Validation
        crnn.train(False)
        running_loss = 0.0
        for batch in val_data_loader:
            images, labels, labels_shifted = batch
            labels = labels.permute(1, 0)
            labels_shifted = labels_shifted.permute(1, 0)
            inputs = (images, labels_shifted) # images should be is BATCH_SIZExCxHxW; shifted_labels SEQ_LENxBATCH_SIZE

            optimizer.zero_grad()
            outputs = crnn(inputs)

            output_seq_numbers = torch.argmax(outputs, dim=2)
            print(f'{"Validation prediction: ".ljust(15)} {output_seq_numbers[:, 0]}')
            print(f'{"Validation real: ".ljust(15)} {labels.permute(1, 0)[0]}')
            print('\n')

            outputs = outputs.view(size=(-1, alphabet_size))
            loss = criterion(outputs, labels.reshape(shape=(-1,)))

            running_loss += loss
        
        print(f'\nAverage validation loss in epoch {epoch}: {running_loss / math.floor(len(val_dataset) / batch_size):.4f}\n')



if __name__ == '__main__':
    #data.generate_scores(num_symbols=50, length_range=(3,5), num_scores=1024)
    train_real()