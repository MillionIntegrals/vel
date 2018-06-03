import waterboy.api.base as base


class StdoutStreaming(base.Callback):
    """ Stream results to stdout """
    def __init__(self, log_frequency, source=None):
        self.log_frequency = log_frequency
        self.source = source

    def on_batch_end(self, progress_idx,  metrics):
        if progress_idx.batch_number % self.log_frequency == 0:
            print('Train Epoch: {:04} [{:06}/{:06} ({:02.0f}%)]\t{}'.format(
                progress_idx.epoch_number,
                progress_idx.batch_number * self.source.batch_size,
                len(self.source.train_source.dataset), 100. * progress_idx.batch_number / len(self.source.train_source),
                self.metrics_string(metrics))
            )

    def on_epoch_end(self, epoch_idx, metrics):
        print(f"=>>>>>>>>>> EPOCH {epoch_idx}")
        print("Train     ", " ".join(["{} {:.06f}".format(k.split(':')[1], metrics[k]) for k in sorted([k for k in metrics.keys() if k.startswith('train:')])]))
        print("Validation", " ".join(["{} {:.06f}".format(k.split(':')[1], metrics[k]) for k in sorted([k for k in metrics.keys() if k.startswith('val:')])]))
        print(f"=>>>>>>>>>> DONE")

    def metrics_string(self, metrics, precision=6):
        """ Return a string describing current values of all metrics """
        return " ".join([("{}: {:." + str(precision) + "f}").format(name, value) for name, value in metrics.items()])


def create(log_frequency=100, source=None):
    """ Waterboy create function """
    return StdoutStreaming(log_frequency=log_frequency, source=source)
