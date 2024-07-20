from monai.data import Dataset, DataLoader as MonaiLoader
from sklearn.model_selection import GroupShuffleSplit
from random import sample


class DataLoader:
    """
    A custom dataloader to split the data into train/val/test 
    """
    def __init__(self, data, valid_split, test_split, seed):
        """
        Splits the dataset into the specified ratios, ensures a unique set of cases per set.
        
        Args:
        data: Tuple: The first element should be a list of dicts containing image paths and targets
                     The second element should be a list of cases identifiers
        valid_split: float: the desired validation split ratio in [0,1]
        test_split: float: the desired test split ratio in [0,1], 
                           the train split is automatically determines as 1-(valid_split+test_split)
        seed: float: the desired seed for shuffling reproducability
        """
        if valid_split == 0 and test_split == 0:
            self.train_data = data[0]
            self.valid_data = []
            self.test_data = []
        else: # split to eval
            gss = GroupShuffleSplit(n_splits=1, test_size=valid_split+test_split, random_state=seed)
            train_indices, eval_indices = next(gss.split(data[0], groups=data[1]))
            self.train_data = [data[0][train_idx] for train_idx in train_indices]
            eval_data = [data[0][i] for i in eval_indices]
            eval_groups = [data[1][i] for i in eval_indices]
            if test_split == 0:
                self.valid_data = eval_data
                self.test_data = []
            elif valid_split == 0:
                self.test_data = eval_data
                self.valid_data = []
            else: # split eval to test and valid
                gss_eval = GroupShuffleSplit(n_splits=1, test_size=valid_split/(valid_split+test_split), random_state=seed)
                test_indices, valid_indices = next(gss_eval.split(eval_data, groups=eval_groups))
                self.valid_data = [eval_data[valid_idx] for valid_idx in valid_indices]
                self.test_data = [eval_data[test_idx] for test_idx in test_indices]

    def trainloader(self, transforms, batch_size, shuffle = True, train_ratio = 1):
        """
        Creates a MONAI DataLoader for the training set
        Args:
        transforms: Compose: a compose object of the training transforms to be applied during loading
        batch_size: int: the number of samples to be loaded per iteration
        shuffle: bool: determines whether to shuffle the data or not. Default is True.
        train_ratio: float: determines the ratio of training data in [0,1]. Default is 1 indicating using the whole training set.
        """
        if train_ratio != 1:
            self.train_data = sample(self.train_data, int(train_ratio * len(self.train_data)))
        dataset = Dataset(self.train_data, transform = transforms)
        dataloader = MonaiLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn=lambda batch: batch)
        return dataloader

    def validloader(self, transforms, batch_size):
        """
        Creates a MONAI DataLoader for the validation set
        Args:
        transforms: Compose: a compose object of the validation transforms to be applied during loading
        batch_size: int: the number of samples to be loaded per iteration
        """
        dataset = Dataset(self.valid_data, transform = transforms)
        dataloader = MonaiLoader(dataset, batch_size = batch_size, 
                                 shuffle = False, collate_fn=lambda batch: batch) # No shuffling for evaluation
        return dataloader
    
    def testloader(self, transforms, batch_size):
        """
        Creates a MONAI DataLoader for the testing set
        Args:
        transforms: Compose: a compose object of the testing transforms to be applied during loading
        batch_size: int: the number of samples to be loaded per iteration
        """
        dataset = Dataset(self.test_data, transform = transforms)
        dataloader = MonaiLoader(dataset, batch_size = batch_size, shuffle = False, collate_fn=lambda batch: batch)
        return dataloader
