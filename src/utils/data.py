import os
import torch
import lightning as L

from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import io

import re

# TODO: Offload to .py file and add instrument selection


class ECallistoDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        # Pfade der Bild-Dateien
        self.file_paths = []
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.endswith(".png"):
                    self.file_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Laden des Bildes
        image = io.read_image(self.file_paths[idx], mode=io.ImageReadMode.GRAY)

        # Anwenden der Transformationen
        if self.transform:
            image = self.transform(image)

        image = image.squeeze()

        # Extrahieren der Klasse aus dem Ordnernamen
        label = os.path.dirname(self.file_paths[idx]).split("/")[-1]
        filename = os.path.basename(self.file_paths[idx])

        # Extrahieren des Instruments aus dem Dateinamen
        instrument = self.get_instrument(filename)

        return image, filename, label, instrument

    def getlabels(self):
        return [
            os.path.dirname(self.file_paths[idx]).split("/")[-1]
            for idx in range(len(self.file_paths))
        ]

    def get_instrument(self, filename):
        pattern = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}_(.*?)_(?:\d|None)")

        # Suchen Sie nach Übereinstimmungen im Dateinamen
        match = pattern.search(filename)
        if match:
            instrument_name = match.group(1)
            return instrument_name
        else:
            return "Unknown" 




class ECallistoDataModule(L.LightningDataModule):
    def __init__(self, data_folder, transform, batch_size, num_workers, val_ratio, test_ratio):
        super().__init__()
        self.data_folder = data_folder
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def setup(self, stage=None):
        dataset = ECallistoDataset(self.data_folder, transform=self.transform)

        # Extract labels
        labels = dataset.getlabels()

        # Split the dataset into train, validation, and test sets while preserving class distribution
        class_indices = {label: [] for label in set(labels)}
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)

        train_indices, val_indices, test_indices = [], [], []
        for indices in class_indices.values():
            num_samples = len(indices)
            num_val_samples = int(num_samples * self.val_ratio)
            num_test_samples = int(num_samples * self.test_ratio)

            val_test_indices = random_split(
                indices,
                [
                    num_val_samples,
                    num_test_samples,
                    num_samples - num_val_samples - num_test_samples,
                ],
            )

            val_indices.extend(val_test_indices[0])
            test_indices.extend(val_test_indices[1])
            train_indices.extend(val_test_indices[2])

        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)
        self.test_dataset = Subset(dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
    
    def get_dataset_instruments(self, dataset_type):
        assert dataset_type in ["train", "val", "test"], "Unbekannter Dataset-Typ"

        # Wählen Sie das richtige Subset basierend auf dem dataset_type
        dataset_subset = None
        if dataset_type == "train":
            dataset_subset = self.train_dataset
        elif dataset_type == "val":
            dataset_subset = self.val_dataset
        elif dataset_type == "test":
            dataset_subset = self.test_dataset

        # Stellen Sie sicher, dass dataset_subset ein Subset-Objekt und kein None ist
        if dataset_subset is None:
            raise ValueError(f"Dataset-Typ {dataset_type} nicht gefunden")

        # Abrufen der Indizes für das aktuelle Subset
        subset_indices = dataset_subset.indices

        # Abrufen der Instrumentennamen für die gegebenen Indizes aus dem zugrunde liegenden Dataset
        instruments = [dataset_subset.dataset.get_instrument(dataset_subset.dataset.file_paths[i]) for i in subset_indices]

        return instruments
