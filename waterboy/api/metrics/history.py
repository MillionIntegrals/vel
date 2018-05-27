import pandas as pd


class TrainingHistory:
    """ Simple aggregator for the training history """
    def __init__(self):
        self.data = []

    def add(self, epoch_result):
        self.data.append(epoch_result)

    def frame(self):
        return pd.DataFrame(self.data).set_index('epoch_idx')
