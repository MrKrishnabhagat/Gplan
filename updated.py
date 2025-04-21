import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import matplotlib.colors as mcolors
from matplotlib.path import Path
import copy
import random
from collections import defaultdict

st.set_page_config(layout="wide")
st.title("📐 Draw & Fill Shapes with Blocks + Adjacencies")

# Initialize session state
if "points" not in st.session_state:
    st.session_state.points = []
if "grid" not in st.session_state:
    st.session_state.grid = None
if "area" not in st.session_state:
    st.session_state.area = None
if "is_closed" not in st.session_state:
    st.session_state.is_closed = False
if "blocks" not in st.session_state:
    st.session_state.blocks = []
if "solution_grid" not in st.session_state:
    st.session_state.solution_grid = None
if "solution_found" not in st.session_state:
    st.session_state.solution_found = False
if "placed_blocks" not in st.session_state:
    st.session_state.placed_blocks = []
if "adjacencies" not in st.session_state:
    st.session_state.adjacencies = []  # List of (block1_id, block2_id) tuples
if "adjacency_count" not in st.session_state:
    st.session_state.adjacency_count = 0  # Total adjacencies in the solution

# Grid settings
GRID_SIZE = 10


# Shoelace area formula
def compute_area(points):
    if len(points) < 3:
        return 0
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    x.append(points[0][0])
    y.append(points[0][1])
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(len(points))))


# Create a grid representation of the shape
def create_grid_from_shape(points, grid_size):
    # Create a grid filled with zeros
    grid = [[0 for _ in range(grid_size + 1)] for _ in range(grid_size + 1)]

    # Fill the grid with 1s for cells inside the polygon
    if len(points) >= 3:
        # Create a polygon path
        path = Path(points)

        # Check each cell if it's inside the polygon
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                # Use the center of the cell for more accurate containment check
                if path.contains_point((j + 0.5, i + 0.5)):
                    grid[grid_size - i][j] = 1

    return grid


# Block placement functions
def can_place(grid, block, row, col):
    # Check if the block can be placed at the given position
    for i in range(len(block)):
        for j in range(len(block[i])):
            if block[i][j] == 1:
                # Check if position is within grid bounds
                if row + i >= len(grid) or col + j >= len(grid[0]):
                    return False
                # Check if position is part of the shape (marked as 1)
                if grid[row + i][col + j] != 1:
                    return False
    return True


def place(grid, block, row, col, block_id):
    for i in range(len(block)):
        for j in range(len(block[i])):
            if block[i][j] == 1:
                grid[row + i][col + j] = (
                    block_id + 2
                )  # +2 to avoid conflict with shape markers


def remove(grid, block, row, col):
    for i in range(len(block)):
        for j in range(len(block[i])):
            if block[i][j] == 1:
                grid[row + i][col + j] = 1  # Reset to 1 (part of shape)


def get_rots(block):
    rotations = []
    curr_rot = block
    seen_rots = set()

    # Try all 4 possible rotations
    for _ in range(4):
        # Convert to tuple of tuples for hashability
        rot_tuple = tuple(tuple(row) for row in curr_rot)
        rot_str = str(rot_tuple)

        if rot_str not in seen_rots:
            seen_rots.add(rot_str)
            rotations.append(curr_rot)

        # Rotate 90 degrees clockwise
        curr_rot = [list(row) for row in zip(*curr_rot[::-1])]

    return rotations


# Check if two blocks are adjacent in the grid
def are_adjacent(grid, block1_id, block2_id):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == block1_id + 2:  # First block found
                # Check all 4 directions
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < len(grid)
                        and 0 <= nj < len(grid[0])
                        and grid[ni][nj] == block2_id + 2
                    ):
                        return True
    return False


# Count total adjacencies in the solution
def count_adjacencies(grid, block_count):
    adjacency_count = 0
    for block1_id in range(block_count):
        for block2_id in range(block1_id + 1, block_count):
            if are_adjacent(grid, block1_id, block2_id):
                adjacency_count += 1
    return adjacency_count


# Check if all required adjacencies are satisfied
def check_adjacency_constraints(grid, adjacencies):
    for block1_id, block2_id in adjacencies:
        if not are_adjacent(grid, block1_id, block2_id):
            return False
    return True


# Try to fit all blocks in the shape with adjacency constraints
def solve_with_adjacencies(grid, blocks, adjacencies, maximize_adjacencies=False):
    best_solution = None
    best_adjacency_count = -1
    attempts = 0
    max_attempts = 2600  # Limit search to avoid long wait times

    while attempts < max_attempts:
        attempts += 1
        # Make a deep copy of the grid for this attempt
        curr_grid = copy.deepcopy(grid)
        used_blocks = []
        placed_block_info = []

        # Try placing each block in a random order
        block_indices = list(range(len(blocks)))
        random.shuffle(block_indices)

        all_placed = True
        for block_idx in block_indices:
            block = blocks[block_idx]
            block_placed = False

            # Try all possible rotations
            rotations = get_rots(block)
            rot_indices = list(range(len(rotations)))
            random.shuffle(rot_indices)

            for rot_idx in rot_indices:
                rot = rotations[rot_idx]

                # Try all positions on the grid in random order
                positions = [
                    (row, col)
                    for row in range(len(grid))
                    for col in range(len(grid[0]))
                ]
                random.shuffle(positions)

                for row, col in positions:
                    if can_place(curr_grid, rot, row, col):
                        # Place this block
                        place(curr_grid, rot, row, col, block_idx)
                        used_blocks.append(block_idx)
                        placed_block_info.append((block_idx, row, col, rot_idx, rot))
                        block_placed = True
                        break

                if block_placed:
                    break

            if not block_placed:
                all_placed = False
                break

        if all_placed:
            # Check if adjacency constraints are satisfied
            constraints_satisfied = check_adjacency_constraints(curr_grid, adjacencies)

            if constraints_satisfied:
                # Count total adjacencies for optimization
                adj_count = count_adjacencies(curr_grid, len(blocks))

                if adj_count > best_adjacency_count:
                    best_adjacency_count = adj_count
                    best_solution = (curr_grid, placed_block_info, adj_count)

                    # If not maximizing adjacencies, return the first valid solution
                    if not maximize_adjacencies:
                        return True, placed_block_info, curr_grid, adj_count

    # Return the best solution found, if any
    if best_solution:
        return True, best_solution[1], best_solution[0], best_solution[2]
    else:
        return False, [], None, 0


# Calculate total cell count in a block
def block_cell_count(block):
    return sum(sum(row) for row in block)


# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Draw Your Shape")
    st.write("Enter coordinates to draw a shape (x, y):")

    # Point input
    x = st.number_input(
        "X coordinate", min_value=0, max_value=GRID_SIZE, step=1, key="x_coord"
    )
    y = st.number_input(
        "Y coordinate", min_value=0, max_value=GRID_SIZE, step=1, key="y_coord"
    )

    # Add point button
    if st.button("Add Point"):
        # Check if the point already exists
        if [x, y] not in st.session_state.points:
            # If there's at least one point, check if we're creating valid horizontal/vertical lines
            if len(st.session_state.points) > 0:
                last_x, last_y = st.session_state.points[-1]
                # Only allow horizontal or vertical lines
                if (x == last_x or y == last_y) and (x != last_x or y != last_y):
                    st.session_state.points.append([x, y])
                else:
                    st.warning("Only horizontal or vertical lines are allowed.")
            else:
                # First point can be added anywhere
                st.session_state.points.append([x, y])

    # Check if shape can be closed
    if len(st.session_state.points) >= 3:
        first_x, first_y = st.session_state.points[0]
        last_x, last_y = st.session_state.points[-1]

        # Enable closing if we can make a vertical or horizontal line back to start
        can_close = (first_x == last_x or first_y == last_y) and (
            first_x != last_x or first_y != last_y
        )

        # Show close shape button only if not already closed
        if not st.session_state.is_closed:
            close_button = st.button("Close Shape", disabled=not can_close)
            if close_button and can_close:
                st.session_state.points.append(
                    [first_x, first_y]
                )  # Add first point to close
                st.session_state.is_closed = True
                # Calculate area
                st.session_state.area = compute_area(
                    st.session_state.points[:-1]
                )  # Exclude last point for calculation

                # Create grid representation
                st.session_state.grid = create_grid_from_shape(
                    st.session_state.points, GRID_SIZE
                )

    # Show points as a table
    if st.session_state.points:
        st.subheader("📍 Points")
        df = pd.DataFrame(st.session_state.points, columns=["X", "Y"])
        df.index = np.arange(1, len(df) + 1)  # Start index from 1
        st.dataframe(df)

    # Display area if shape is closed
    if st.session_state.is_closed and st.session_state.area is not None:
        st.success(
            f"✅ Closed figure detected — Area: **{st.session_state.area:.2f} square units**"
        )

    # Block creation section
    if st.session_state.is_closed:
        st.subheader("2. Add Blocks to Fill the Shape")

        col_len, col_width = st.columns(2)
        with col_len:
            block_length = st.number_input(
                "Length", min_value=1, max_value=GRID_SIZE, value=1
            )
        with col_width:
            block_width = st.number_input(
                "Width", min_value=1, max_value=GRID_SIZE, value=1
            )

        if st.button("Add Block"):
            # Create the block as a 2D array of 1s
            block = [[1 for _ in range(block_width)] for _ in range(block_length)]
            st.session_state.blocks.append(block)
            st.success(f"Added block with dimensions {block_length} x {block_width}")

        # List of blocks
        if st.session_state.blocks:
            st.subheader("📦 Your Blocks")
            for i, block in enumerate(st.session_state.blocks):
                st.write(f"Block {i+1}: {len(block)} x {len(block[0])}")

        # Adjacency constraints section
        if len(st.session_state.blocks) >= 2:
            st.subheader("3. Define Block Adjacencies")

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                block1 = st.selectbox(
                    "Block 1",
                    options=list(range(1, len(st.session_state.blocks) + 1)),
                    format_func=lambda x: f"Block {x}",
                )
            with col_b2:
                block2 = st.selectbox(
                    "Block 2",
                    options=list(range(1, len(st.session_state.blocks) + 1)),
                    format_func=lambda x: f"Block {x}",
                )

            if st.button("Add Adjacency"):
                if block1 != block2:
                    # Convert to 0-indexed
                    adj = (block1 - 1, block2 - 1)
                    # Store as tuple with smaller index first for consistency
                    adj = (min(adj), max(adj))

                    # Check if this adjacency already exists
                    if adj not in st.session_state.adjacencies:
                        st.session_state.adjacencies.append(adj)
                        st.success(
                            f"Added adjacency between Block {block1} and Block {block2}"
                        )
                    else:
                        st.warning("This adjacency already exists!")
                else:
                    st.warning("Cannot add adjacency between the same block!")

            # Show adjacencies list
            if st.session_state.adjacencies:
                st.subheader("Required Adjacencies:")
                for b1, b2 in st.session_state.adjacencies:
                    st.write(f"Block {b1+1} must be adjacent to Block {b2+1}")

                if st.button("Clear All Adjacencies"):
                    st.session_state.adjacencies = []
                    st.success("All adjacencies cleared")

        # Solve button
        if st.session_state.blocks:
            st.subheader("4. Solve and Fill")

            # Calculate total area of blocks
            total_cells = sum(
                block_cell_count(block) for block in st.session_state.blocks
            )
            shape_area = (
                st.session_state.area if st.session_state.area is not None else 0
            )

            # Display block area info
            st.info(f"Total block area: {total_cells} square units")
            if total_cells > shape_area:
                st.warning(
                    f"Warning: Total block area ({total_cells}) exceeds shape area ({shape_area:.2f})"
                )

            col_solve, col_maximize = st.columns(2)

            with col_solve:
                solve_button = st.button("Satisfy Adjacencies")
            with col_maximize:
                maximize_button = st.button("Maximize Adjacencies")

            if solve_button or maximize_button:
                # Create a copy of the grid for solving
                if st.session_state.grid:
                    with st.spinner("Finding solution..."):
                        success, placed_blocks, solution_grid, adj_count = (
                            solve_with_adjacencies(
                                st.session_state.grid,
                                st.session_state.blocks,
                                st.session_state.adjacencies,
                                maximize_adjacencies=maximize_button,
                            )
                        )

                        st.session_state.solution_found = success
                        st.session_state.placed_blocks = placed_blocks
                        st.session_state.adjacency_count = adj_count

                        if success:
                            st.session_state.solution_grid = solution_grid
                            st.success(
                                f"✅ Solution found! All {len(st.session_state.blocks)} blocks placed successfully."
                            )
                            st.info(f"Total adjacencies in solution: {adj_count}")

                            # Count required adjacencies satisfied
                            req_adj_satisfied = sum(
                                1
                                for adj in st.session_state.adjacencies
                                if are_adjacent(solution_grid, adj[0], adj[1])
                            )

                            st.success(
                                f"Required adjacencies satisfied: {req_adj_satisfied}/{len(st.session_state.adjacencies)}"
                            )
                        else:
                            st.error(
                                "❌ Could not find a solution. Try different block sizes or fewer adjacency constraints."
                            )

        # Reset button
        if st.button("Reset Everything"):
            st.session_state.points = []
            st.session_state.grid = None
            st.session_state.area = None
            st.session_state.is_closed = False
            st.session_state.blocks = []
            st.session_state.solution_grid = None
            st.session_state.solution_found = False
            st.session_state.placed_blocks = []
            st.session_state.adjacencies = []
            st.session_state.adjacency_count = 0
            st.rerun()

# Visualization column
with col2:
    st.subheader("Shape Visualization")

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set up the plot
    ax.set_xlim(-0.5, GRID_SIZE + 0.5)
    ax.set_ylim(-0.5, GRID_SIZE + 0.5)
    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))
    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)

    # Draw the shape
    if len(st.session_state.points) > 0:
        # Draw lines between points
        points_array = np.array(st.session_state.points)
        ax.plot(points_array[:, 0], points_array[:, 1], "b-", linewidth=2)

        # Draw and label each point
        for i, (x, y) in enumerate(st.session_state.points):
            ax.plot(x, y, "ro", markersize=8)
            ax.text(x + 0.1, y + 0.1, f"({x},{y})", fontsize=9)

        # Shade the polygon if closed
        if st.session_state.is_closed and len(st.session_state.points) > 3:
            # Create a polygon
            polygon = Polygon(points_array[:-1], alpha=0.2, color="lightgray")
            ax.add_patch(polygon)

    # Show solution if available
    if st.session_state.solution_found and st.session_state.solution_grid is not None:
        grid = st.session_state.solution_grid
        # Use a colormap with more distinct colors
        cmap = plt.cm.get_cmap("tab20", len(st.session_state.blocks))

        # Draw filled blocks
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] > 1:  # Block ID (+2)
                    block_id = grid[i][j] - 2
                    color = cmap(block_id)
                    ax.add_patch(
                        Rectangle((j, GRID_SIZE - i), 1, 1, color=color, alpha=0.7)
                    )
                    ax.text(
                        j + 0.5,
                        GRID_SIZE - i + 0.5,
                        str(block_id + 1),  # Display 1-based index
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                    )

        # Highlight adjacencies
        if st.session_state.adjacencies and st.checkbox(
            "Highlight Adjacencies", value=True
        ):
            # Find all pairs of adjacent cells from different blocks
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if grid[i][j] > 1:  # Found a block cell
                        block_id = grid[i][j] - 2
                        # Check all 4 directions
                        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                        for di, dj in directions:
                            ni, nj = i + di, j + dj
                            if (
                                0 <= ni < len(grid)
                                and 0 <= nj < len(grid[0])
                                and grid[ni][nj] > 1
                                and grid[ni][nj] != grid[i][j]
                            ):
                                # Found an adjacency between different blocks
                                neighbor_id = grid[ni][nj] - 2
                                # Only draw once per pair (when block_id < neighbor_id)
                                if block_id < neighbor_id:
                                    # Check if this is a required adjacency
                                    is_required = (
                                        block_id,
                                        neighbor_id,
                                    ) in st.session_state.adjacencies
                                    # Draw a thicker line for required adjacencies
                                    line_width = 4 if is_required else 2
                                    line_style = "-" if is_required else "--"
                                    line_color = "green" if is_required else "black"

                                    # Calculate midpoints for the line
                                    mid_i = (i + ni) / 2
                                    mid_j = (j + nj) / 2

                                    # Draw line between block centers
                                    if di == 0:  # Horizontal adjacency
                                        ax.plot(
                                            [j + 0.5, nj + 0.5],
                                            [GRID_SIZE - i + 0.5, GRID_SIZE - ni + 0.5],
                                            line_style,
                                            color=line_color,
                                            linewidth=line_width,
                                        )
                                    else:  # Vertical adjacency
                                        ax.plot(
                                            [j + 0.5, nj + 0.5],
                                            [GRID_SIZE - i + 0.5, GRID_SIZE - ni + 0.5],
                                            line_style,
                                            color=line_color,
                                            linewidth=line_width,
                                        )

    # Display the plot
    st.pyplot(fig)

    # Display grid for debugging
    if st.session_state.grid is not None and st.checkbox("Show Grid Representation"):
        st.write("Grid representation of shape (1 = inside shape, 0 = outside):")
        grid_df = pd.DataFrame(st.session_state.grid)
        st.dataframe(grid_df)

    # Display adjacency matrix if solution found
    if st.session_state.solution_found and st.session_state.placed_blocks:
        if st.checkbox("Show Adjacency Matrix"):
            st.subheader("Block Adjacency Matrix")

            # Create adjacency matrix
            num_blocks = len(st.session_state.blocks)
            adj_matrix = [[0 for _ in range(num_blocks)] for _ in range(num_blocks)]

            # Fill adjacency matrix
            for i in range(num_blocks):
                for j in range(i + 1, num_blocks):
                    if are_adjacent(st.session_state.solution_grid, i, j):
                        adj_matrix[i][j] = adj_matrix[j][i] = 1

            # Convert to DataFrame for better display
            adj_df = pd.DataFrame(
                adj_matrix,
                index=[f"Block {i+1}" for i in range(num_blocks)],
                columns=[f"Block {i+1}" for i in range(num_blocks)],
            )
            st.dataframe(adj_df)

    # Display placement details if solution found
    if st.session_state.solution_found and st.session_state.placed_blocks:
        st.subheader("Block Placement Details")
        for block_id, row, col, rot_idx, _ in st.session_state.placed_blocks:
            actual_block = st.session_state.blocks[block_id]
            st.write(
                f"Block {block_id+1} ({len(actual_block)}x{len(actual_block[0])}) placed at position ({col},{GRID_SIZE-row}) with rotation {rot_idx*90}°"
            )
