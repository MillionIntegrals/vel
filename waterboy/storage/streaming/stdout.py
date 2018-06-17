import waterboy.api.base as base


class StdoutStreaming(base.Callback):
    """ Stream results to stdout """
    def on_epoch_end(self, epoch_idx, metrics):
        print(f"=>>>>>>>>>> EPOCH {epoch_idx.global_epoch_idx}")
        print("Train     ", " ".join(["{} {:.06f}".format(k.split(':')[1], metrics[k]) for k in sorted([k for k in metrics.keys() if k.startswith('train:')])]))
        print("Validation", " ".join(["{} {:.06f}".format(k.split(':')[1], metrics[k]) for k in sorted([k for k in metrics.keys() if k.startswith('val:')])]))
        print(f"=>>>>>>>>>> DONE")

    def metrics_string(self, metrics, precision=6):
        """ Return a string describing current values of all metrics """
        return " ".join([("{}: {:." + str(precision) + "f}").format(name, value) for name, value in metrics.items()])


def create():
    """ Waterboy create function """
    return StdoutStreaming()
