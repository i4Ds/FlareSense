import os
import math
import torch
import torchvision
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class ECallistoDataset(Dataset):
    def __init__(
        self,
        metadata,
        data_folder="",
        img_size=(224, 224),
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.img_size = img_size
        self.data_folder = data_folder

    def __len__(self):
        return len(self.metadata)

    def __preprocess(self, image, length):
        length_s = torch.tensor((image.index.max() - image.index.min()).total_seconds())
        resampling_s = torch.round((length_s / length)).int().item()
        image = image.resample(f"{resampling_s}s").max()
        image = image.interpolate(method="linear", limit_direction="both")
        return image.values.T

    def __getitem__(self, idx):
        spectra_metadata = {key: str(value) for key, value in self.metadata.iloc[idx].to_dict().items()}
        spectra_metadata["label"] = spectra_metadata["type"]

        image = pd.read_parquet(os.path.join(self.data_folder, spectra_metadata["file_name"]))
        image = self.__preprocess(image, length=self.img_size[0])
        image = torch.tensor(image).float()
        image = image.unsqueeze(0)

        if torch.isnan(image).all():
            raise ValueError(f"All values in image are NaN: {spectra_metadata['file_name']}")

        if torch.isnan(image).any():
            raise ValueError(f"NaN values in image: {spectra_metadata['file_name']}")

        image = torchvision.transforms.functional.resize(image, self.img_size, antialias=True)

        return image, spectra_metadata


class ECallistoDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_folder,
        batch_size,
        num_workers,
        val_ratio,
        test_ratio,
        img_size,
        use_augmented_data=True,
        filter_instruments=[],
        seed=0,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.img_size = img_size
        self.use_augmented_data = use_augmented_data
        self.filter_instruments = filter_instruments
        self.seed = seed

        self.metadata = pd.read_csv(
            os.path.join(self.data_folder, "metadata.csv"), parse_dates=["datetime_start", "datetime_end"]
        )
        
    def __remove_augmented_data(self, df):
        df["folder_name"] = df["file_name"].str.split("/").str[-2]
        df = df.groupby("folder_name").sample(1)
        df = df.sort_index()
        df = df.reset_index(drop=True)
        df = df.drop(columns=["folder_name"])
        return df

    def setup(self, stage=None):
        # filter out where data is under 14 minutes and over 16 minutes
        self.metadata = self.metadata.loc[
            (self.metadata.datetime_end - self.metadata.datetime_start).dt.total_seconds() >= 14 * 60
        ]
        self.metadata = self.metadata.loc[
            (self.metadata.datetime_end - self.metadata.datetime_start).dt.total_seconds() <= 16 * 60
        ]

        # filter instruments
        if self.filter_instruments != []:
            self.metadata = self.metadata.loc[self.metadata.instruments.isin(self.filter_instruments)]

        # split by folder name
        observations = self.metadata.file_name.str.split("/").str[-2].unique()
        num_samples = len(observations)
        num_val_samples = int(num_samples * self.val_ratio)
        num_test_samples = int(num_samples * self.test_ratio)

        # create train, val, test split
        torch.manual_seed(self.seed)
        indices = random_split(
            observations,
            [
                num_samples - num_val_samples - num_test_samples,
                num_val_samples,
                num_test_samples,
            ],
        )

        train_observations = observations[indices[0].indices]
        val_observations = observations[indices[1].indices]
        test_observations = observations[indices[2].indices]

        train_metadata = self.metadata[self.metadata.file_name.str.split("/").str[-2].isin(train_observations)].copy()
        val_metadata = self.metadata[self.metadata.file_name.str.split("/").str[-2].isin(val_observations)].copy()
        test_metadata = self.metadata[self.metadata.file_name.str.split("/").str[-2].isin(test_observations)].copy()

        # filter augmentated data (only use files named "1.parquet") # may have bias
        if not self.use_augmented_data:
            train_metadata = self.__remove_augmented_data(train_metadata)
        
        # filter all augmentations in val and test set, since they are not representative
        val_metadata = self.__remove_augmented_data(val_metadata)
        test_metadata = self.__remove_augmented_data(test_metadata)

        # create datasets
        self.train_dataset = ECallistoDataset(train_metadata, data_folder=self.data_folder, img_size=self.img_size)
        self.val_dataset = ECallistoDataset(val_metadata, data_folder=self.data_folder, img_size=self.img_size)
        self.test_dataset = ECallistoDataset(test_metadata, data_folder=self.data_folder, img_size=self.img_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
