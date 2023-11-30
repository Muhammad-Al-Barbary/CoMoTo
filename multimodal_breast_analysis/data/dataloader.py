from monai.data import Dataset, DataLoader as MonaiLoader
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data, test_split, seed):
        self.train_data, self.test_data = train_test_split(data, test_size=test_split, random_state=seed)
         
    def trainloader(self, transforms, batch_size, shuffle):
        dataset = Dataset(self.train_data, transform = transforms)
        dataloader = MonaiLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn=lambda batch: batch)
        return dataloader
    
    def testloader(self, transforms, batch_size, shuffle):
        dataset = Dataset(self.test_data, transform = transforms)
        dataloader = MonaiLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn=lambda batch: batch)
        return dataloader
