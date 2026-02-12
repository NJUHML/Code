from typing import Dict, Tuple
from torchmetrics import MetricCollection
from rich.table import Table


def print_result(metric_collection: Dict[str, MetricCollection], metric_names: Tuple[str, ...], lead_times: Tuple[int, ...]):

    results = {}

    for lead_time, m_collection in metric_collection.items():
        results[lead_time] = m_collection.compute()

    # create table
    table = Table(show_header=True, header_style="bold magenta")

    # col
    table.add_column("Lead time", style="cyan", justify="center")

    for metric_name in metric_names:
        table.add_column(metric_name.upper(), justify="center", style="green")

    # data fill in each row
    all_values = {metric: [] for metric in metric_names}

    for h in lead_times:
        row = [str(h)]
        
        for metric_name in metric_names:
            value = results[f"lead_{h}"][metric_name].item()
            
            all_values[metric_name].append(value)
            
            row.append(f" {value:8f} ")
            
        table.add_row(*row)

    table.add_section()

    # mean value over all lead times
    mean_row = ["Mean value"]

    for metric_name in metric_names:
        mean_value = sum(all_values[metric_name]) / len(all_values[metric_name])
        
        mean_row.append(f" {mean_value:8f} ")

    table.add_row(*mean_row, style="bold yellow")

    return table