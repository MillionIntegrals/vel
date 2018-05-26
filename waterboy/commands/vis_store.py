import visdom
import itertools as it


def _column_original_name(name):
    """ Return original name of the metric """
    if ':' in name:
        return name.split(':')[-1]
    else:
        return name


def visdom_push_metrics(vis, metrics):
    """ Push metrics to visdom """
    visdom_send_metrics(vis, metrics, 'replace')


def visdom_send_metrics(vis, metrics, update=None):
    """ Append metrics to visdom """
    for name, groups in it.groupby(metrics.columns, key=_column_original_name):
        groups = list(groups)

        for idx, group in enumerate(groups):
            if vis.win_exists(name):
                update = update
            else:
                update = None

            vis.line(
                metrics[group].values,
                metrics.index.values,
                win=name,
                name=group,
                opts={
                    'title': name,
                    'showlegend': True
                },
                update=update
            )

            if name != group:
                if vis.win_exists(group):
                    update = update
                else:
                    update = None

                vis.line(
                    metrics[group].values,
                    metrics.index.values,
                    win=group,
                    name=group,
                    opts={
                        'title': group,
                        'showlegend': True
                    },
                    update=update
                )


def visdom_append_metrics(vis, metrics, first_epoch=False):
    """ Append metrics to visdom """
    visited = {}

    for name, groups in it.groupby(metrics.columns, key=_column_original_name):
        groups = list(groups)

        for group in groups:
            if vis.win_exists(name) and (not visited.get(group, False)) and first_epoch:
                update = 'replace'
            elif not vis.win_exists(name):
                update = None
            else:
                update = 'append'

            vis.line(
                metrics[group].values,
                metrics.index.values,
                win=name,
                name=group,
                opts={
                    'title': name,
                    'showlegend': True
                },
                update=update
            )

            if name != group:
                if vis.win_exists(group) and first_epoch:
                    update = 'replace'
                elif not vis.win_exists(group):
                    update = None
                else:
                    update = 'append'

                vis.line(
                    metrics[group].values,
                    metrics.index.values,
                    win=group,
                    name=group,
                    opts={
                        'title': group,
                        'showlegend': True
                    },
                    update=update
                )


class VisdomCommand:
    def __init__(self, model_config, storage):
        self.model_config = model_config
        self.storage = storage
        self.vis = visdom.Visdom(env=self.model_config.run_name)

    def run(self):
        metrics = self.storage.get_metrics().drop('run_name', axis=1)
        visdom_push_metrics(self.vis, metrics)


def create(model_config, storage):
    return VisdomCommand(model_config, storage)
