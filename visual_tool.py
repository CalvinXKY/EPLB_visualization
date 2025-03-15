import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def visualize_4d_array(arr):
    """
    Draw a hierarchical diagram of a 4D array.

    Parameters:
        arr: 4D array with shape [dim0, dim1, dim2, dim3]
    """
    dim0 = len(arr)
    dim1 = len(arr[0]) if dim0 > 0 else 0
    dim2 = len(arr[0][0]) if dim1 > 0 else 0
    dim3 = len(arr[0][0][0]) if dim2 > 0 else 0

    # Count value frequencies per row
    value_counts_per_row = {}
    for i in range(dim0):
        value_counts = {}
        for j in range(dim1):
            for k in range(dim2):
                for l in range(dim3):
                    value = arr[i][j][k][l]
                    if value in value_counts:
                        value_counts[value] += 1
                    else:
                        value_counts[value] = 1
        value_counts_per_row[i] = value_counts

    # Assign colors to repeated values per row
    repeated_values_per_row = {}
    color_maps_per_row = {}
    for i in range(dim0):
        value_counts = value_counts_per_row[i]
        repeated_values = {value: idx for idx, value in enumerate([k for k, v in value_counts.items() if v > 1])}
        num_repeated = len(repeated_values)
        if num_repeated > 0:
            cmap = plt.cm.tab20
            norm = plt.Normalize(vmin=0, vmax=num_repeated - 1)
            color_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        else:
            color_map = None
        repeated_values_per_row[i] = repeated_values
        color_maps_per_row[i] = color_map

    # Set figure size
    plt.figure(figsize=(15, 6))

    # Calculate layout parameters
    total_width = 12
    cell_width = total_width / dim1
    cell_height = 0.8
    block_width = cell_width * 0.8 / dim2
    block_height = cell_height * 0.8
    sub_block_width = block_width * 0.8 / dim3
    sub_block_height = block_height * 0.8

    # Draw grid
    for i in range(dim0):
        for j in range(dim1):
            x = j * cell_width
            y = (dim0 - 1 - i) * cell_height
            rect = patches.Rectangle((x, y), cell_width * 0.9, cell_height * 0.9,
                                     facecolor='lightgray', edgecolor='black', alpha=0.3)
            plt.gca().add_patch(rect)

            if j == 0:
                label_x = x - cell_width * 0.1
                label_y = y + cell_height * 0.5
                plt.text(label_x, label_y, f'Layer {i}', ha='right', va='center', fontsize=20, color='black',
                         rotation=90)

            if i == dim0 - 1:
                label_x = x + cell_width * 0.5
                label_y = y - cell_height * 0.1
                plt.text(label_x, label_y, f'Node {j}', ha='center', va='top', fontsize=20, color='black')

            for k in range(dim2):
                block_x = x + k * block_width + cell_width * 0.05
                block_y = y + cell_height * 0.05

                rect_block = patches.Rectangle((block_x, block_y), block_width * 0.9, block_height * 0.9,
                                               facecolor='none', edgecolor='black', alpha=0.3,
                                               linestyle='dashed')
                plt.gca().add_patch(rect_block)

                plt.text(block_x + block_width * 0.1, block_y + block_height * 0.9,
                         f'GPU {k}', ha='left', va='top', fontsize=8, color='black')

                for l in range(dim3):
                    sub_block_x = block_x + l * sub_block_width + block_width * 0.05
                    sub_block_y = block_y + block_height * 0.05

                    value = arr[i][j][k][l]

                    repeated_values = repeated_values_per_row[i]
                    color_map = color_maps_per_row[i]

                    if value in repeated_values:
                        if color_map:
                            color = color_map.to_rgba(repeated_values[value])
                        else:
                            color = 'gray'
                    else:
                        color = 'none'

                    rect_sub_block = patches.Rectangle((sub_block_x, sub_block_y), sub_block_width * 0.9,
                                                       sub_block_height * 0.9,
                                                       facecolor=color, edgecolor='black', alpha=0.7)
                    plt.gca().add_patch(rect_sub_block)

                    plt.text(sub_block_x + sub_block_width * 0.45, sub_block_y + sub_block_height * 0.45,
                             f'{value}', ha='center', va='center', fontsize=16)

    for i in range(dim0 - 1):
        y_pos = (dim0 - 1 - i) * cell_height
        x_start = (0 * cell_width) - cell_width * 0.1
        plt.plot([x_start, total_width], [y_pos, y_pos], linestyle='dashed', color='black')

    plt.xlim(-0.5, total_width)
    plt.ylim(-0.5, dim0 * cell_height + 0.5)
    plt.axis('off')
    plt.title('EPLB Visualization     ---(@kaiyuan)')

    plt.tight_layout()
    plt.show()

def visualize_ep_inputs(weight):
    """
    Visualize experts inputs
    weight: [layers, num_logical_experts], the load statistics for all logical experts
    """
    # Determine array dimensions

    if not isinstance(weight, np.ndarray):
      weight = np.array(weight)
    dim0, dim1 = weight.shape

    # Handle empty array case
    if dim0 == 0 or dim1 == 0:
        print("Empty array provided.")
        return

    # Compute color mapping
    values = weight.flatten()
    max_val = np.max(values)
    min_val = np.min(values)

    # Set color map
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    color_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Set up plot
    plt.figure(figsize=(15, 6))
    ax = plt.subplot(111)

    # Calculate layout parameters
    total_width = 12
    cell_width = total_width / dim1
    cell_height = 1.2

    # Draw each element
    for i in range(dim0):
        for j in range(dim1):
            x = j * cell_width
            y = (dim0 - 1 - i) * cell_height

            # Draw rectangle
            rect = patches.Rectangle((x, y), cell_width * 0.9, cell_height * 0.9,
                                     facecolor=color_map.to_rgba(weight[i][j]),
                                     edgecolor='black', alpha=0.7)
            ax.add_patch(rect)

            # Add text with larger font and black color
            text_x = x + cell_width * 0.45
            text_y = y + cell_height * 0.45
            ax.text(text_x, text_y, f'EP_{j}: {weight[i][j]}',
                    ha='center', va='center', fontsize=10, color='black')

            # Add layer label in first column
            if j == 0:
                label_x = x - cell_width * 0.1
                label_y = y + cell_height * 0.5
                ax.text(label_x, label_y, f'Layer {i}',
                        ha='right', va='center', fontsize=12, color='black', rotation=90)

        # Add dashed lines between layers
        if i < dim0 - 1:
            y_pos = (dim0 - 1 - i) * cell_height
            x_start = -cell_width * 0.1
            x_end = total_width
            ax.plot([x_start, x_end], [y_pos, y_pos], linestyle='dashed', color='black')

    # Set axis limits and style
    ax.set_xlim(-0.5, total_width)
    ax.set_ylim(-0.5, dim0 * cell_height + 0.5)
    ax.axis('off')
    ax.set_title('EP Weights Visualization     ---(@kaiyuan)')

    # Add colorbar
    plt.colorbar(color_map, ax=ax, label='Value')

    plt.tight_layout()
    plt.show()

def reshape_map(phy2log, num_nodes, num_gpus):
  np_phy2log = np.array(phy2log, dtype=int)
  np_phy2log = np_phy2log.reshape(np_phy2log.shape[0], num_nodes, int(num_gpus/num_nodes), int(np_phy2log.shape[-1]/num_gpus))
  return np_phy2log
