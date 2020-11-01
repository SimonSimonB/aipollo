from torch import log_
import torchvision
import cv2
import torch.optim 
import torch
import math

import models
import transforms
import data
import datetime
import pathlib

def predict_test():
    # Load image
    pass

def write_dataset_to_disk():
    data.write_to_disk(512, 512, [[9]])
    #data.write_to_disk(512, 512, [[36, 38]])

def train_net():
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    weight_foreground_factor = None
    batch_size = 4
    validation_interval = 1500
    save_interval = 1500
    class_groups = [[-1]] # 34 = black notehead; 36 = half notehead; 38 = whole notehead


    model = models.UNet()
    log_path = pathlib.Path('./aipollo_processor/detectors/unet_torch/logs/' + '-'.join([str(class_group) for class_group in class_groups]) + '--' + datetime.datetime.now().strftime('%Y-%m-%d-%H.%M.%S'))
    model.load_state_dict(torch.load('aipollo_processor/detectors/unet_torch/logs/[-1]--2020-10-23-18.07.47/7500.pt'))

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    dataset = data.ScoreSnippetsDataset(
        IMAGE_HEIGHT, 
        IMAGE_WIDTH, 
        class_groups, 
        transform=torchvision.transforms.Compose([transforms.RandomResize(), transforms.Noise(), transforms.Normalize()])
    )

    train_dataset, val_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[math.floor(len(dataset)*0.9), len(dataset) - math.floor(len(dataset)*0.9)])
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0, drop_last=True)
    print(f'#instances in train / validation set: {len(train_dataset)} / {len(val_dataset)}')

    instances_seen = 0
    for epoch in range(1024):
        print(f'Epoch: {epoch}' )

        # Training
        for batch in train_data_loader:
            model.train(True)

            images, labels = batch

            optimizer.zero_grad()
            outputs = model(images)

            outputs_predictions_in_last_dimension = outputs.permute(0, 2, 3, 1)
            outputs_flattened = torch.reshape(outputs_predictions_in_last_dimension, (-1,))
            labels_flattened = torch.flatten(labels)

            if weight_foreground_factor:
                pixel_weights = labels_flattened + (1.0 / (weight_foreground_factor - 1))
                pixel_weights *= 1.0 / pixel_weights.mean()
                criterion.weight = pixel_weights

            loss = criterion(outputs_flattened, labels_flattened)
            loss.backward()
            optimizer.step()
        
            # Debug show a prediction
            cv2.imshow('Input', images[0][0].numpy())
            cv2.waitKey(1)
            cv2.imshow('Targets', labels[0].numpy())
            cv2.waitKey(1)
            first_prediction = outputs[0][0]
            cv2.imshow('Prediction', first_prediction.detach().numpy())
            cv2.waitKey(1)

            print(loss)

            instances_seen += batch_size
            if instances_seen % ((validation_interval // batch_size) * batch_size) == 0:
                # Print validation loss.
                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    for batch in val_data_loader:
                        images, labels = batch

                        outputs = model(images)

                        outputs_predictions_in_last_dimension = outputs.permute(0, 2, 3, 1)
                        outputs_flattened = torch.reshape(outputs_predictions_in_last_dimension, (-1,))
                        labels_flattened = torch.flatten(labels)

                        if weight_foreground_factor:
                            pixel_weights = labels_flattened + (1.0 / (weight_foreground_factor - 1))
                            pixel_weights *= 1.0 / pixel_weights.mean()
                            criterion.weight = pixel_weights

                        total_loss += criterion(outputs_flattened, labels_flattened)

                    print(f'The validation loss after {instances_seen} is {total_loss}')

                    log_path.mkdir(parents=True, exist_ok=True)
                    with open(str(log_path / 'validation_error.txt'), 'a') as f:
                        f.write(f'The validation loss after {instances_seen} instances seen is {total_loss}\n')
            

            if instances_seen % ((save_interval // batch_size) * batch_size) == 0:
                log_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(log_path / f'{instances_seen}.pt'))

                

if __name__ == '__main__':
    #cProfile.run('train_net()')
    #write_dataset_to_disk()
    train_net()

