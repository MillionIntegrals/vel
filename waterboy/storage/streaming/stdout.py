import waterboy.api.base as base


class StdoutStreaming(base.Callback):
    """ Stream results to stdout """
    def on_epoch_end(self, epoch_idx, metrics):
        print(f"=>>>>>>>>>> EPOCH {epoch_idx.global_epoch_idx}")

        if any(':' not in x for x in metrics.keys()):
            print("Metrics     ", " ".join(["{} {:.06f}".format(k, metrics[k]) for k in sorted([k for k in metrics.keys() if ':' not in k])]))

        if any(x.startswith('train:') for x in metrics.keys()):
            print("Train     ", " ".join(["{} {:.06f}".format(k.split(':')[1], metrics[k]) for k in sorted([k for k in metrics.keys() if k.startswith('train:')])]))

        if any(x.startswith('val:') for x in metrics.keys()):
            print("Validation", " ".join(["{} {:.06f}".format(k.split(':')[1], metrics[k]) for k in sorted([k for k in metrics.keys() if k.startswith('val:')])]))

        print(f"=>>>>>>>>>> DONE")


def create():
    """ Waterboy create function """
    return StdoutStreaming()
