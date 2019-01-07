from vel.api import EpochInfo, Callback


class StdoutStreaming(Callback):
    """ Stream results to stdout """
    def on_epoch_end(self, epoch_info: EpochInfo):
        if epoch_info.training_info.run_name:
            print(f"=>>>>>>>>>> EPOCH {epoch_info.global_epoch_idx} [{epoch_info.training_info.run_name}]")
        else:
            print(f"=>>>>>>>>>> EPOCH {epoch_info.global_epoch_idx}")

        if any(':' not in x for x in epoch_info.result.keys()):
            self._print_metrics_line(epoch_info.result, head=None)

        head_set = sorted({x.split(':')[0] + ':' for x in epoch_info.result.keys() if ':' in x})

        for head in head_set:
            if any(x.startswith(head) for x in epoch_info.result.keys()):
                self._print_metrics_line(epoch_info.result, head)

        print(f"=>>>>>>>>>> DONE")

    @staticmethod
    def _print_metrics_line(metrics, head=None):
        if head is None:
            head = 'Metrics:'

            metrics_list = [
                "{} {:.06f}".format(k, metrics[k])
                for k in sorted([k for k in metrics.keys() if ':' not in k])
            ]
        else:
            metrics_list = [
                "{} {:.06f}".format(k.split(':')[1], metrics[k])
                for k in sorted([k for k in metrics.keys() if k.startswith(head)])
            ]

        print('{0: <10}'.format(head.capitalize()), " ".join(metrics_list))


def create():
    """ Vel factory function """
    return StdoutStreaming()
