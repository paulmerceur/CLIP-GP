import os
import pickle
import math
import random
from collections import defaultdict

from utils.dataset_base import DATASET_REGISTRY, Datum, DatasetBase, mkdir_if_missing


@DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):

    dataset_dir = "oxford_pets"

    def __init__(self, config):
        root = os.path.abspath(os.path.expanduser(config.dataset.root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            trainval = self.read_data(split_file="trainval.txt")
            test = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = config.dataset.num_shots
        if num_shots >= 1:
            seed = config.seed
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = getattr(config.dataset, 'subsample_classes', 'all')
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based label
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    def split_trainval(self, trainval):
        # Randomly split a dataset, each class may have
        # different number of instances
        per_class = defaultdict(list)
        for item in trainval:
            per_class[item.label].append(item)
        train, val = [], []
        for label, items in per_class.items():
            n_total = len(items)
            n_train = round(n_total * 0.5)
            assert n_train > 0
            random.shuffle(items)
            train.extend(items[:n_train])
            val.extend(items[n_train:])
        return train, val

    @staticmethod
    def save_split(train, val, test, split_path, path_prefix = ""):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        DatasetBase.write_json(split, split_path)
        print(f"Saved split to {split_path}")

    @staticmethod
    def read_split(split_path: str, path_prefix: str = ""):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=str(classname))
                out.append(item)
            return out

        print(f"Reading split from {split_path}")
        split = DatasetBase.read_json(split_path)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])
        return train, val, test
    
    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)
        
        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # the first half
        else:
            selected = labels[m:]  # the second half
        
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label in selected:
                    item_new = Datum(
                        impath=item.impath,
                        label=relabeler[item.label],
                        classname=item.classname
                    )
                    dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output
