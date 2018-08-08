import torch.utils.data as data


class DataFlow(data.Dataset):
    """ A dataset wrapping underlying data source with transformations """
    def __init__(self, dataset, transformations, tag):
        self.dataset = dataset

        if transformations is None:
            self.transformations = []
        else:
            self.transformations = [t for t in transformations if tag in t.tags]

        self.tag = tag

    def get_raw(self, index):
        return self.dataset[index]

    def __getitem__(self, index):
        raw_x, raw_y = self.dataset[index]

        for t in self.transformations:
            if t.mode == 'x':
                raw_x = t(raw_x)
            elif t.mode == 'y':
                raw_y = t(raw_y)
            elif t.mode == 'both':
                raw_x, raw_y = t(raw_x, raw_y)
            else:
                raise RuntimeError(f"Mode {t.mode} not recognized")

        return raw_x, raw_y

    def denormalize(self, datum, mode='x'):
        for t in self.transformations[::-1]:
            if t.mode == mode:
                datum = t.denormalize(datum)

        return datum

    def __len__(self):
        return len(self.dataset)
