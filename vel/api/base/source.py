import torch.utils.data as data
import vel.api.data as wdata


class Source:
    """ Very simple wrapper for a training and validation datasource """
    def __init__(self, train_source, val_source, num_workers, batch_size, augmentations=None):
        self.train_source = train_source
        self.val_source = val_source

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.augmentations = augmentations
        # self.tta = test_time_augmentation

        # Derived values
        self.train_ds = wdata.DataFlow(self.train_source, augmentations, tag='train')
        self.val_ds = wdata.DataFlow(self.val_source, augmentations, tag='val')

        self.train_loader = data.DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.val_loader = data.DataLoader(
            self.val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        # self.val_tta_loader = self.val_loader
        # else:
        #     self.val_tta_loader = self.tta.loader(
        #         self.val_source, self.augmentations, self.batch_size, self.num_workers
        #     )
            
    # def is_tta_enabled(self):
    #     """ Is test-time augmentation enabled """
    #     return self.tta is not None
    #
    # def tta_accumulator(self, result_accumulator):
    #     """ Return metric accumulator for the 'tta' calculations """
    #     return self.tta.accumulator(result_accumulator, self.val_source)

    def train_dataset(self):
        """ Return the training dataset """
        return self.train_ds

    def val_dataset(self):
        """ Return the validation dataset """
        return self.val_ds

    def train_iterations_per_epoch(self):
        """ Return number of iterations per epoch """
        return len(self.train_loader)

    def val_iterations_per_epoch(self):
        """ Return number of iterations per epoch - validation """
        return len(self.val_loader)

    # def tta_postprocess(self, x):
    #     """ Prostprocess the test-time-augmentation data """
    #     raise NotImplementedError
