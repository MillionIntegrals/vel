import attr
import itertools as it


@attr.s(auto_attribs=True)
class VisdomSettings:
    """ Settings for connecting to the visdom server """
    stream_lr: bool = False
    server: str = 'http://localhost'
    endpoint: str = 'events'
    port: int = 8097


def _column_original_name(name):
    """ Return original name of the metric """
    if ':' in name:
        return name.split(':')[-1]
    else:
        return name


def visdom_push_metrics(vis, metrics):
    """ Push metrics to visdom """
    visdom_send_metrics(vis, metrics, 'replace')


def visdom_send_metrics(vis, metrics, update='replace'):
    """ Send set of metrics to visdom """
    visited = {}

    sorted_metrics = sorted(metrics.columns, key=_column_original_name)
    for metric_basename, metric_list in it.groupby(sorted_metrics, key=_column_original_name):
        metric_list = list(metric_list)

        for metric in metric_list:
            if vis.win_exists(metric_basename) and (not visited.get(metric, False)):
                update = update
            elif not vis.win_exists(metric_basename):
                update = None
            else:
                update = 'append'

            vis.line(
                metrics[metric].values,
                metrics.index.values,
                win=metric_basename,
                name=metric,
                opts={
                    'title': metric_basename,
                    'showlegend': True
                },
                update=update
            )

            if metric_basename != metric and len(metric_list) > 1:
                if vis.win_exists(metric):
                    update = update
                else:
                    update = None

                vis.line(
                    metrics[metric].values,
                    metrics.index.values,
                    win=metric,
                    name=metric,
                    opts={
                        'title': metric,
                        'showlegend': True
                    },
                    update=update
                )


def visdom_append_metrics(vis, metrics, first_epoch=False):
    """ Append metrics to visdom """
    visited = {}

    sorted_metrics = sorted(metrics.columns, key=_column_original_name)
    for metric_basename, metric_list in it.groupby(sorted_metrics, key=_column_original_name):
        metric_list = list(metric_list)

        for metric in metric_list:
            if vis.win_exists(metric_basename) and (not visited.get(metric, False)) and first_epoch:
                update = 'replace'
            elif not vis.win_exists(metric_basename):
                update = None
            else:
                update = 'append'

            vis.line(
                metrics[metric].values,
                metrics.index.values,
                win=metric_basename,
                name=metric,
                opts={
                    'title': metric_basename,
                    'showlegend': True
                },
                update=update
            )

            if metric_basename != metric and len(metric_list) > 1:
                if vis.win_exists(metric) and first_epoch:
                    update = 'replace'
                elif not vis.win_exists(metric):
                    update = None
                else:
                    update = 'append'

                vis.line(
                    metrics[metric].values,
                    metrics.index.values,
                    win=metric,
                    name=metric,
                    opts={
                        'title': metric,
                        'showlegend': True
                    },
                    update=update
                )

