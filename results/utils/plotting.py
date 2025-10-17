import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attack_comparison(model_attack_data, directory, cmap="viridis"):
    dfs = []
    for model, attacks in model_attack_data.items():
        match = re.search(r'(\d+(?:\.\d+)?)[bB]', model)
        model_size = float(match.group(1)) if match else 0

        # Extract model family (everything before the first size number)
        family_match = re.match(r'([a-zA-Z\-]+)', model)
        model_family = family_match.group(1) if family_match else model
        for attack, metrics in attacks.items():
            success_rate = (metrics['successful'] / metrics['total']) * 100.0 if metrics['total'] > 0 else 0.0
            dfs.append({
                'Model': model,
                'Model Family': model_family,
                'Model Size': model_size,
                'Model Size Label': f"{model_size}B",
                'Attack': attack,
                'Success Rate (%)': success_rate
            })

    df = pd.DataFrame(dfs)
    df = df.sort_values(['Attack', 'Model Family', 'Model Size'])

    for attack_type in df['Attack'].unique():
        fig, ax = plt.subplots(figsize=(10, 5))  # <-- capture figure and axes
        subset = df[df['Attack'] == attack_type]
        palette = sns.color_palette(cmap, n_colors=subset['Model Family'].nunique())

        # Group by model family to connect points
        for i, (family_name, family_data) in enumerate(subset.groupby('Model Family')):
            family_data = family_data.sort_values('Model Size')
            plt.plot(
                family_data['Model Size'],
                family_data['Success Rate (%)'],
                marker='o',
                label=family_name,
                color=palette[i],
                linewidth=2
            )

        plt.title(f"{attack_type.replace('.csv','')} Success Rate by Model Family", pad=15)
        plt.ylim(0, 110)
        plt.ylabel('Success Rate (%)')
        plt.xlabel('Model Size')
        plt.xticks(
            ticks=sorted(subset['Model Size'].unique()),
            labels=[f"{x}B" for x in sorted(subset['Model Size'].unique())]
        )
        plt.legend(title='Model Family', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        os.chdir(directory)
        fig.savefig(attack_type.split('.')[0] +".png", dpi=300, bbox_inches="tight")
        plt.show()




def plot_size_comparison(result_dataframes, extract_means, directory, palette):

    model_data = {}
    for name, df in result_dataframes.items():
        match = re.search(r'(\d+(?:\.\d+)?)[bB]', name)
        size_float = float(match.group(1)) if match else 0
        size = f"{size_float}B"

        if 'qwen' in name.lower():
            family = 'qwen'
        elif 'llama' in name.lower():
            family = 'llama'
        else:
            family = 'unknown'

        model_data[f"{family}_{size}"] = {
            'family': family,
            'size': size,
            'size_num': size_float,
            'metrics': extract_means(df)
        }

    # Group by metric groups
    first_model = next(iter(model_data.values()))
    metric_groups = {k: v for k, v in first_model['metrics'].items() if v}

    n_rows = len(metric_groups)
    n_cols = max(len(metrics) for metrics in metric_groups.values())
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Get unique families and sizes for plotting
    families = sorted(set(data['family'] for data in model_data.values()))

    # Sort sizes numerically using the stored size_num
    all_sizes = sorted(set((data['size'], data['size_num']) for data in model_data.values()),
                       key=lambda x: x[1])
    all_size_labels = [size[0] for size in all_sizes]  # ['1B', '3B', '4B', '8B', '12B']
    all_size_nums = [size[1] for size in all_sizes]    # [1, 3, 4, 8, 12]

    for row_idx, (group_name, metrics) in enumerate(metric_groups.items()):
        for col_idx, (metric_name, _) in enumerate(metrics.items()):
            ax = axes[row_idx][col_idx]

            # Prepare data for each family
            family_data = {}
            for family in families:
                family_values = []
                family_sizes_num = []
                family_size_labels = []

                for size_label, size_num in zip(all_size_labels, all_size_nums):
                    key = f"{family}_{size_label}"
                    if key in model_data:
                        metrics_dict = model_data[key]['metrics']
                        if (group_name in metrics_dict and
                                metric_name in metrics_dict[group_name]):
                            value = metrics_dict[group_name][metric_name]
                            family_values.append(value)
                            family_sizes_num.append(size_num)
                            family_size_labels.append(size_label)

                if family_values:
                    family_data[family] = {
                        'sizes_num': family_sizes_num,
                        'size_labels': family_size_labels,
                        'values': family_values
                    }

            # Plot each family
            colors = sns.color_palette(palette, len(families))
            for i, (family, data_dict) in enumerate(family_data.items()):
                ax.plot(data_dict['sizes_num'], data_dict['values'],
                        marker='o', linewidth=2, markersize=8,
                        color=colors[i], label=family.title(),
                        markerfacecolor='white', markeredgecolor=colors[i],
                        markeredgewidth=2)

                # Annotations
                for size_num, value, size_label in zip(data_dict['sizes_num'],
                                                      data_dict['values'],
                                                      data_dict['size_labels']):
                    ax.text(size_num, value + 0.01,
                            f"{value:.2f}", ha='center', va='bottom',
                            fontsize=9, color=colors[i], fontweight='bold')

            # Set ticks, labels, and titles
            ax.set_xticks(all_size_nums)
            ax.set_xticklabels(all_size_labels)
            ax.set_title(f"{group_name.replace('_', ' ').title()}\n{metric_name.title().split('.')[0]}",
                         fontsize=12, fontweight='bold')

            # Y limits
            all_values = []
            for data_dict in family_data.values():
                all_values.extend(data_dict['values'])

            if all_values:
                max_val = max(all_values)
                y_upper = 1.1 if max_val <= 1 else max_val * 1.15
                ax.set_ylim(0, y_upper)

            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, linestyle='--')
            if all_values:
                ax.legend(loc='lower right' if max_val <= 1 else 'upper right')

            if row_idx == n_rows - 1:
                ax.set_xlabel('Model Size', fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel('Score', fontweight='bold')

            # Save each subplot as standalone PNG
            title = f"{group_name}_{metric_name.split('.')[0]}"
            single_fig, single_ax = plt.subplots(figsize=(5, 4))

            # Re-plot into new figure to preserve labels/legend
            for line in ax.get_lines():
                single_ax.plot(line.get_xdata(), line.get_ydata(),
                               marker=line.get_marker(),
                               linestyle=line.get_linestyle(),
                               linewidth=line.get_linewidth(),
                               markersize=line.get_markersize(),
                               color=line.get_color(),
                               label=line.get_label(),
                               markerfacecolor=line.get_markerfacecolor(),
                               markeredgecolor=line.get_markeredgecolor(),
                               markeredgewidth=line.get_markeredgewidth())

            # Copy style
            single_ax.set_xticks(ax.get_xticks())
            single_ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()])
            single_ax.set_ylim(ax.get_ylim())
            single_ax.set_title(ax.get_title(), fontsize=12, fontweight='bold')
            single_ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
            single_ax.set_ylabel(ax.get_ylabel(), fontweight='bold')
            single_ax.grid(True, alpha=0.3, linestyle='--')
            if ax.get_legend():
                single_ax.legend(loc=ax.get_legend()._loc)

            single_fig.tight_layout()
            single_fig.savefig(os.path.join(directory, f"{title}.png"),
                               dpi=300, bbox_inches="tight")
            plt.close(single_fig)

    plt.tight_layout()
    os.chdir(directory)
    fig.savefig("size_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
