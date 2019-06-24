from vel.api import EpochInfo, Callback


class StdoutStreaming(Callback):
    """ Stream results to stdout """
    def on_epoch_end(self, epoch_info: EpochInfo):
        if epoch_info.training_info.run_name:
            print(f"=>>>>>>>>>> EPOCH {epoch_info.global_epoch_idx} [{epoch_info.training_info.run_name}]")
        else:
            print(f"=>>>>>>>>>> EPOCH {epoch_info.global_epoch_idx}")

        if any(x.dataset is None for x in epoch_info.result.keys()):
            self._print_metrics_line(epoch_info.result, dataset=None)

        head_set = sorted({x.dataset for x in epoch_info.result.keys() if x.dataset is not None})

        for head in head_set:
            self._print_metrics_line(epoch_info.result, head)

        print(f"=>>>>>>>>>> DONE")

    @staticmethod
    def _print_metrics_line(metrics, dataset=None):
        if dataset is None:
            dataset = 'Metrics:'

            metrics_list = [
                "{}/{} {:.06f}".format(k.scope, k.name, metrics[k])
                for k in sorted([k for k in metrics.keys() if k.dataset is None])
            ]
        else:
            metrics_list = [
                "{}/{} {:.06f}".format(k.scope, k.name, metrics[k])
                for k in sorted([k for k in metrics.keys() if k.dataset == dataset])
            ]

        print('{0: <10}'.format(dataset.capitalize()), " ".join(metrics_list))


def create():
    """ Vel factory function """
    return StdoutStreaming()
