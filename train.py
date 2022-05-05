import time
import numpy as np
import torch
import torch.utils.data as data
from sklearn import metrics
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, Compose
from torchvision import transforms
from tqdm import tqdm
import wandb
from MRData import MRData
from Net import Net
from config import config
import os
def get_data(task):
    # Define the Augmentation here only
    augments = Compose([
        # Convert the image to Tensor
        transforms.Lambda(lambda x: torch.Tensor(x)),
        # Randomly rotate the image with an angle
        # between -25 degrees to 25 degrees
        RandomRotate(25),
        # Randomly translate the image by 11% of
        # image height and width
        RandomTranslate([0.11, 0.11]),
        # Randomly flip the image
        RandomFlip(),
        # Change the order of image channels
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    ])

    print('Loading Train Dataset of {} task...'.format(task))
    # Load training dataset
    train_data = MRData(task, train=True, transform=augments)
    train_loader = data.DataLoader(
        train_data, batch_size=1, num_workers=0, shuffle=True
    )

    print('Loading Validation Dataset of {} task...'.format(task))
    # Load validation dataset
    val_data = MRData(task, train=False)
    val_loader = data.DataLoader(
        val_data, batch_size=1, num_workers=0, shuffle=False
    )

    return train_loader, val_loader, train_data.weights, val_data.weights


def _train_model(model, train_loader, epoch, num_epochs, optimizer, criterion, current_lr, log_every=100):
    # Set to train mode
    model.train()

    # Initialize the predicted probabilities
    y_probs = []
    # Initialize the groundtruth labels
    y_gt = []
    # Initialize the loss between the groundtruth label
    # and the predicted probability
    losses = []

    # Iterate over the training dataset
    for i, (images, label) in enumerate(tqdm(train_loader)):
        # Reset the gradient by zeroing it
        optimizer.zero_grad()

        # If GPU is available, transfer the images and label
        # to the GPU
        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()

        # Obtain the prediction using the model
        output = model(images)

        # Evaluate the loss by comparing the prediction
        # and groundtruth label
        loss = criterion(output, label)
        # Perform a backward propagation
        loss.backward()
        # Modify the weights based on the error gradient
        optimizer.step()

        # Add current loss to the list of losses
        loss_value = loss.item()
        losses.append(loss_value)

        # Find probabilities from output using sigmoid function
        probas = torch.sigmoid(output)

        # Add current groundtruth label to the list of groundtruths
        y_gt.append(int(label.item()))
        # Add current probabilities to the list of probabilities
        y_probs.append(probas.item())

        try:
            # Try finding the area under ROC curve
            auc = metrics.roc_auc_score(y_gt, y_probs)
        except:
            # Use default value of area under ROC curve as 0.5
            auc = 0.5

        if (i % log_every == 0) & (i > 0):
            # Display the information about average training loss and area under ROC curve
            print('''[Epoch: {0} / {1} | Batch : {2} / {3} ]| Avg Train Loss {4} | Train AUC : {5} | lr : {6}'''.
                format(
                epoch + 1,
                num_epochs,
                i,
                len(train_loader),
                np.round(np.mean(losses), 4),
                np.round(auc, 4),
                current_lr
            )
            )

    # Find mean area under ROC curve and training loss
    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)

    return train_loss_epoch, train_auc_epoch


def _evaluate_model(model, val_loader, criterion, epoch, num_epochs, current_lr, log_every=20):
    """Runs model over val dataset and returns auc and avg val loss"""

    # Set to eval mode
    model.eval()
    # List of probabilities obtained from the model
    y_probs = []
    # List of groundtruth labels
    y_gt = []
    # List of losses obtained
    losses = []

    # Iterate over the validation dataset
    for i, (images, label) in enumerate(val_loader):
        # If GPU is available, load the images and label
        # on GPU
        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()

        # Obtain the model output by passing the images as input
        output = model(images)
        # Evaluate the loss by comparing the output and groundtruth label
        loss = criterion(output, label)
        # Add loss to the list of losses
        loss_value = loss.item()
        losses.append(loss_value)
        # Find probability for each class by applying
        # sigmoid function on model output
        probas = torch.sigmoid(output)
        # Add the groundtruth to the list of groundtruths
        y_gt.append(int(label.item()))
        # Add predicted probability to the list
        y_probs.append(probas.item())

        try:
            # Evaluate area under ROC curve based on the groundtruth label
            # and predicted probability
            auc = metrics.roc_auc_score(y_gt, y_probs)
        except:
            # Default area under ROC curve
            auc = 0.5

        if (i % log_every == 0) & (i > 0):
            # Display the information about average validation loss and area under ROC curve
            print('''[Epoch: {0} / {1} | Batch : {2} / {3} ]| Avg Val Loss {4} | Val AUC : {5} | lr : {6}'''.
                format(
                epoch + 1,
                num_epochs,
                i,
                len(val_loader),
                np.round(np.mean(losses), 4),
                np.round(auc, 4),
                current_lr
            )
            )

    # Find mean area under ROC curve and validation loss
    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)

    return val_loss_epoch, val_auc_epoch


def train(config: dict):
    """
    Function where actual training takes place
    Args:
        config (dict) : Configuration to train with
    """

    print('Starting to Train Model...')

    train_loader, val_loader, train_wts, val_wts = get_data(config['task'])

    print('Initializing Model...')
    model = Net()
    if torch.cuda.is_available():
        model = model.cuda()
        train_wts = train_wts.cuda()
        val_wts = val_wts.cuda()

    print('Initializing Loss Method...')
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    val_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=val_wts)

    if torch.cuda.is_available():
        criterion = criterion.cuda()
        val_criterion = val_criterion.cuda()

    print('Setup the Optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)

    starting_epoch = config['starting_epoch']
    num_epochs = config['max_epoch']
    log_train = config['log_train']
    log_val = config['log_val']

    best_val_auc = float(0)

    print('Starting Training')

    wandb.log({"lr": config['lr'], "task": config['task']})
    t_start_training = time.time()

    for epoch in tqdm(range(starting_epoch, num_epochs)):

        current_lr = _get_lr(optimizer)
        epoch_start_time = time.time()  # timer for entire epoch

        train_loss, train_auc = _train_model(
            model, train_loader, epoch, num_epochs, optimizer, criterion, current_lr, log_train)

        val_loss, val_auc = _evaluate_model(
            model, val_loader, val_criterion, epoch, num_epochs, current_lr, log_val)

        wandb.log({'Train/AUC_epoch': train_auc, "epoc": epoch})
        wandb.log({'Train/Loss_epoch': train_loss, "epoc": epoch})

        scheduler.step(val_loss)

        t_end = time.time()
        delta = t_end - epoch_start_time

        print("train loss : {0} | train auc {1} | val loss {2} | val auc {3} | elapsed time {4} s".format(
            train_loss, train_auc, val_loss, val_auc, delta))

        print('-' * 30)

        if val_auc > best_val_auc:
            best_val_auc = val_auc

        if bool(config['save_model']):
            file_name = config["out_dir"]+'model_{}_{}_val_auc_{:0.4f}_train_auc_{:0.4f}_epoch_{}.pkl'.format(config['exp_name'],
                                                                                            config['task'], val_auc,
                                                                                            train_auc, epoch + 1)
            torch.save(model, file_name)

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')


def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    os.environ["WANDB_API_KEY"] = config["YOUR_API_KEY"]
    os.environ["WANDB_START_METHOD"] = "thread"
    wandb.init(project="my_MRNET")
    print('Training Configuration')
    print(config)

    train(config=config)

    print('Training Ended...')
