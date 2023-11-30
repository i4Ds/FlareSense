import os
import math
import torch
import torchvision
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class ECallistoDataset(Dataset):
    def __init__(self, observations, transform=None):
        self.observations = observations.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        metadata = {key: str(value) for key, value in self.observations.iloc[idx].to_dict().items()}

        image = torchvision.io.read_image(metadata["file_path"], mode=torchvision.io.ImageReadMode.GRAY)
        if self.transform:
            image = self.transform(image)

        return image, metadata


class ECallistoDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_folder,
        transform,
        batch_size,
        num_workers,
        val_ratio,
        test_ratio,
        min_factor_val_test=2 / 3,
        max_factor_val_test=3 / 2,
        noburst_to_burst_ratio=1,
        split_by_date=False,
        filter_instruments=[],
    ):
        super().__init__()
        self.data_folder = data_folder
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_factor_val_test = min_factor_val_test
        self.max_factor_val_test = max_factor_val_test
        self.noburst_to_burst_ratio = noburst_to_burst_ratio
        self.split_by_date = split_by_date
        self.filter_instruments = filter_instruments

    def __get_dataframe(self, data_folder):
        # Pfade der Bild-Dateien einlesen
        file_paths = []
        for root, _, files in os.walk(data_folder):
            file_paths.extend(os.path.join(root, file) for file in files if file.endswith(".png"))

        # Erstellen eines DataFrames
        observations = pd.DataFrame({"file_path": file_paths})
        observations["file_name"] = observations["file_path"].apply(lambda x: os.path.basename(x))
        observations["start"] = observations["file_name"].apply(lambda x: x.split("_")[0])
        observations["start"] = pd.to_datetime(observations["start"], format="%Y-%m-%d %H-%M-%S")
        observations["end"] = observations["file_name"].apply(lambda x: x.split("_")[1])
        observations["end"] = pd.to_datetime(observations["end"], format="%Y-%m-%d %H-%M-%S")
        observations["instrument"] = observations["file_name"].apply(lambda x: x.split("_")[2:5])
        observations["instrument"] = observations["instrument"].apply(lambda x: "_".join(x))
        observations["instrument"] = observations["instrument"].apply(lambda x: x.replace("_None", ""))
        observations["label"] = observations["file_path"].apply(lambda x: os.path.basename(os.path.dirname(x)))
        observations = observations.drop(columns="file_name")

        return observations

    def setup(self, stage=None):
        self.observations = self.__get_dataframe(self.data_folder)

        # split by date
        if self.split_by_date:
            dates = self.observations["start"].dt.date.unique()

            num_samples = len(dates)
            num_val_samples = int(num_samples * self.val_ratio)
            num_test_samples = int(num_samples * self.test_ratio)

            # if the imbalance is too big, try again
            while True:
                torch.manual_seed(4)
                indices = random_split(
                    dates,
                    [
                        num_samples - num_val_samples - num_test_samples,
                        num_val_samples,
                        num_test_samples,
                    ],
                )

                train_dates = dates[indices[0].indices]
                val_dates = dates[indices[1].indices]
                test_dates = dates[indices[2].indices]

                train_indices = self.observations["start"].dt.date.isin(train_dates)
                val_indices = self.observations["start"].dt.date.isin(val_dates)
                test_indices = self.observations["start"].dt.date.isin(test_dates)

                # filter instruments
                if len(self.filter_instruments):
                    train_indices = train_indices & self.observations["instrument"].isin(self.filter_instruments)
                    val_indices = val_indices & self.observations["instrument"].isin(self.filter_instruments)
                    test_indices = test_indices & self.observations["instrument"].isin(self.filter_instruments)

                # Check if the distribution of bursts is more or less balanced

                total_num_bursts = sum(self.observations["label"] != "no_burst")
                val_num_bursts = sum(self.observations[val_indices]["label"] != "no_burst")
                test_num_bursts = sum(self.observations[test_indices]["label"] != "no_burst")
                min_val_bursts = math.ceil((self.val_ratio * total_num_bursts) * self.min_factor_val_test)
                min_test_bursts = math.ceil((self.test_ratio * total_num_bursts) * self.min_factor_val_test)
                max_val_bursts = math.floor((self.val_ratio * total_num_bursts) * self.max_factor_val_test)
                max_test_bursts = math.floor((self.test_ratio * total_num_bursts) * self.max_factor_val_test)

                continue_loop = False
                if val_num_bursts < min_val_bursts:
                    print(f"Not enough validation bursts ({val_num_bursts}), minimum {min_val_bursts} needed")
                    continue_loop = True
                if test_num_bursts < min_test_bursts:
                    print(f"Not enough test bursts ({test_num_bursts}), minimum {min_test_bursts} needed")
                    continue_loop = True
                if val_num_bursts > max_val_bursts:
                    print(f"Too many validation bursts ({val_num_bursts}), maximum {max_val_bursts} allowed")
                    continue_loop = True
                if test_num_bursts > max_test_bursts:
                    print(f"Too many test bursts ({test_num_bursts}), maximum {max_test_bursts} allowed")
                    continue_loop = True
                if not continue_loop:
                    print("Dataset split successfully")
                    print(f"Train:\t\t{total_num_bursts - val_num_bursts - test_num_bursts} bursts")
                    print(f"Validation:\t{val_num_bursts} bursts")
                    print(f"Test:\t\t{test_num_bursts} bursts")
                    break

                print("Reshuffling...\n")

            train_dataset = self.observations[train_indices]
            val_dataset = self.observations[val_indices]
            test_dataset = self.observations[test_indices]

            # Anpassung der Anzahl No-Bursts an Burst-Anzahl
            if self.noburst_to_burst_ratio != float('inf'):
                bursts = train_dataset[train_dataset["label"] != "no_burst"]
                nobursts = train_dataset[train_dataset["label"] == "no_burst"]
                nobursts = nobursts.sample(
                    n=math.ceil(self.noburst_to_burst_ratio * len(bursts)),
                    replace=False,
                )
                train_dataset = pd.concat([bursts, nobursts], ignore_index=True)

            # create datasets
            self.train_dataset = ECallistoDataset(train_dataset, transform=self.transform)
            self.val_dataset = ECallistoDataset(val_dataset, transform=self.transform)
            self.test_dataset = ECallistoDataset(test_dataset, transform=self.transform)

        # dont split by date, stratify by label
        else:
            train_indices, val_indices, test_indices = [], [], []
            for label in self.observations["label"].unique():
                label_indices = self.observations["label"] == label
                label_indices = label_indices[label_indices].index

                num_samples = len(label_indices)
                num_val_samples = int(num_samples * self.val_ratio)
                num_test_samples = int(num_samples * self.test_ratio)

                torch.manual_seed(4)
                indices = random_split(
                    label_indices,
                    [
                        num_val_samples,
                        num_test_samples,
                        num_samples - num_val_samples - num_test_samples,
                    ],
                )

                val_indices.extend(label_indices[indices[0].indices])
                test_indices.extend(label_indices[indices[1].indices])
                train_indices.extend(label_indices[indices[2].indices])

            train_dataset = self.observations.iloc[train_indices]
            val_dataset = self.observations.iloc[val_indices]
            test_dataset = self.observations.iloc[test_indices]

            # filter instruments
            if len(self.filter_instruments):
                train_dataset = train_dataset[train_dataset["instrument"].isin(self.filter_instruments)]
                val_dataset = val_dataset[val_dataset["instrument"].isin(self.filter_instruments)]
                test_dataset = test_dataset[test_dataset["instrument"].isin(self.filter_instruments)]

            # Anpassung der Anzahl No-Bursts an Burst-Anzahl
            if self.noburst_to_burst_ratio != float('inf'):
                bursts = train_dataset[train_dataset["label"] != "no_burst"]
                nobursts = train_dataset[train_dataset["label"] == "no_burst"]
                nobursts = nobursts.sample(
                    n=math.ceil(self.noburst_to_burst_ratio * len(bursts)),
                    replace=False,
                )
                train_dataset = pd.concat([bursts, nobursts], ignore_index=True)

            # create datasets
            self.train_dataset = ECallistoDataset(train_dataset, transform=self.transform)
            self.val_dataset = ECallistoDataset(val_dataset, transform=self.transform)
            self.test_dataset = ECallistoDataset(test_dataset, transform=self.transform)

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
