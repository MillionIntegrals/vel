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

    for metric_basename, metric_list in it.groupby(metrics.columns, key=_column_original_name):
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

