import torch.utils.data as data


class Source:
    """ Source of data for supervised learning algorithms """
    def __init__(self):
        pass

    def train_loader(self):
        """ PyTorch loader of training data """
        raise NotImplementedError

    def val_loader(self):
        """ PyTorch loader of validation data """
        raise NotImplementedError

    def train_dataset(self):
        """ Return the training dataset """
        raise NotImplementedError

    def val_dataset(self):
        """ Return the validation dataset """
        raise NotImplementedError

    def train_iterations_per_epoch(self):
        """ Return number of iterations per epoch """
        raise NotImplementedError

    def val_iterations_per_epoch(self):
        """ Return number of iterations per epoch - validation """
        raise NotImplementedError


class TextData(Source):
    """ An NLP torchtext data source """
    def __init__(self, train_source, val_source, train_iterator, val_iterator, data_field, target_field):
        super().__init__()

        self.train_source = train_source
        self.val_source = val_source
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.data_field = data_field
        self.target_field = target_field

    def train_loader(self):
        """ PyTorch loader of training data """
        return self.train_iterator

    def val_loader(self):
        """ PyTorch loader of validation data """
        return self.val_iterator

    def train_dataset(self):
        """ Return the training dataset """
        return self.train_source

    def val_dataset(self):
        """ Return the validation dataset """
        return self.val_source

    def train_iterations_per_epoch(self):
        """ Return number of iterations per epoch """
        return len(self.train_iterator)

    def val_iterations_per_epoch(self):
        """ Return number of iterations per epoch - validation """
        return len(self.val_iterator)


class TrainingData(Source):
    """ Most common source of data combining a basic datasource and sampler """
    def __init__(self, train_source, val_source, num_workers, batch_size, augmentations=None):
        import vel.api.data as vel_data

        super().__init__()

        self.train_source = train_source
        self.val_source = val_source

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.augmentations = augmentations

        # Derived values
        self.train_ds = vel_data.DataFlow(self.train_source, augmentations, tag='train')
        self.val_ds = vel_data.DataFlow(self.val_source, augmentations, tag='val')

        self._train_loader = data.DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self._val_loader = data.DataLoader(
            self.val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    def train_loader(self):
        """ PyTorch loader of training data """
        return self._train_loader

    def val_loader(self):
        """ PyTorch loader of validation data """
        return self._val_loader

    def train_dataset(self):
        """ Return the training dataset """
        return self.train_ds

    def val_dataset(self):
        """ Return the validation dataset """
        return self.val_ds

    def train_iterations_per_epoch(self):
        """ Return number of iterations per epoch """
        return len(self._train_loader)

    def val_iterations_per_epoch(self):
        """ Return number of iterations per epoch - validation """
        return len(self._val_loader)
