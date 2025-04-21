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
import multiprocessing
import concurrent.futures
import time
import heapq
from functools import lru_cache

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
if "solving_progress" not in st.session_state:
    st.session_state.solving_progress = 0

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


# Create a grid representation of the shape - OPTIMIZED with direct path check
def create_grid_from_shape(points, grid_size):
    # Create a grid filled with zeros
    grid = np.zeros((grid_size + 1, grid_size + 1), dtype=np.int8)

    # Fill the grid with 1s for cells inside the polygon
    if len(points) >= 3:
        # Create a polygon path once
        path = Path(points)

        # Vectorized approach for faster checking
        y, x = np.mgrid[0 : grid_size + 1, 0 : grid_size + 1]
        centers = np.vstack([x.ravel() + 0.5, y.ravel() + 0.5]).T

        # Check all points at once
        mask = path.contains_points(centers).reshape(grid_size + 1, grid_size + 1)
        grid[mask] = 1

        # Flip the grid to match the expected orientation
        grid = np.flipud(grid)

    return grid


# Block placement functions - OPTIMIZED
def can_place(grid, block, row, col):
    # Vectorized approach
    height, width = len(block), len(block[0])

    # Quick boundary check
    if row + height > grid.shape[0] or col + width > grid.shape[1]:
        return False

    # Convert block to numpy array if it's not already
    if not isinstance(block, np.ndarray):
        block = np.array(block, dtype=np.int8)

    # Extract the region from the grid
    region = grid[row : row + height, col : col + width]

    # Check if all block cells (where block==1) can be placed (grid==1)
    # This means for each position where block is 1, grid must also be 1
    return np.all((block == 1) <= (region == 1))


def place(grid, block, row, col, block_id):
    height, width = len(block), len(block[0])

    # Convert block to numpy array if needed
    if not isinstance(block, np.ndarray):
        block = np.array(block, dtype=np.int8)

    # Create a mask for the block cells
    mask = block == 1

    # Apply the mask to update only the cells where block has 1s
    grid[row : row + height, col : col + width][mask] = block_id + 2


def remove(grid, block, row, col):
    height, width = len(block), len(block[0])

    # Convert block to numpy array if needed
    if not isinstance(block, np.ndarray):
        block = np.array(block, dtype=np.int8)

    # Create a mask for the block cells
    mask = block == 1

    # Reset only the cells where block has 1s back to 1
    grid[row : row + height, col : col + width][mask] = 1


# Cache rotations to avoid recalculating
@lru_cache(maxsize=1000)
def get_rots_cached(block_tuple):
    # Convert tuple format back to list for processing
    block = [list(row) for row in block_tuple]
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


def get_rots(block):
    # Convert block to a tuple for caching
    block_tuple = tuple(tuple(row) for row in block)
    return get_rots_cached(block_tuple)


# OPTIMIZED: More efficient adjacency checking
def build_adjacency_map(grid):
    """Build a map of block positions for faster adjacency checks"""
    block_positions = defaultdict(list)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            block_id = grid[i, j]
            if block_id > 1:  # It's a block
                block_positions[block_id].append((i, j))

    return block_positions


def are_adjacent_fast(block_positions, block1_id, block2_id):
    """Fast adjacency check using precomputed positions"""
    # Add 2 to block IDs since grid values are block_id + 2
    block1_id_grid = block1_id + 2
    block2_id_grid = block2_id + 2

    # Get all positions for both blocks
    positions1 = block_positions[block1_id_grid]
    positions2 = block_positions[block2_id_grid]

    # Check each position in block1 against each in block2
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for i, j in positions1:
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if (ni, nj) in positions2:
                return True

    return False


def count_adjacencies_fast(grid, block_count):
    """Count adjacencies using the faster method"""
    block_positions = build_adjacency_map(grid)
    adjacency_count = 0

    for block1_id in range(block_count):
        for block2_id in range(block1_id + 1, block_count):
            if are_adjacent_fast(block_positions, block1_id, block2_id):
                adjacency_count += 1

    return adjacency_count


def check_adjacency_constraints_fast(grid, adjacencies):
    """Check if required adjacencies are satisfied using the faster method"""
    block_positions = build_adjacency_map(grid)

    for block1_id, block2_id in adjacencies:
        if not are_adjacent_fast(block_positions, block1_id, block2_id):
            return False

    return True


# OPTIMIZED: Calculate block area once
def block_cell_count(block):
    """Count the number of cells in a block"""
    if isinstance(block, np.ndarray):
        return np.sum(block)
    return sum(sum(row) for row in block)


# MAJOR OPTIMIZATION: Intelligent backtracking algorithm with heuristics
def solve_with_backtracking(
    grid, blocks, adjacencies, maximize_adjacencies=False, time_limit=30
):
    """Solve using backtracking with heuristics and time limit"""
    # Convert grid to numpy if not already
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid, dtype=np.int8)

    # Start time
    start_time = time.time()

    # Convert blocks to numpy arrays
    blocks = [
        np.array(block, dtype=np.int8) if not isinstance(block, np.ndarray) else block
        for block in blocks
    ]

    # Calculate block areas
    block_areas = [block_cell_count(block) for block in blocks]

    # Sort blocks by area (descending) for better placement
    block_order = sorted(range(len(blocks)), key=lambda i: block_areas[i], reverse=True)

    # Calculate all valid placements for each block
    valid_placements = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Pre-calculate valid positions for each block
    status_text.text("Calculating valid positions...")
    for block_idx in block_order:
        block = blocks[block_idx]
        rotations = get_rots(block)
        block_placements = []

        for rot_idx, rot in enumerate(rotations):
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    if can_place(grid, rot, row, col):
                        block_placements.append((row, col, rot_idx, rot))

        valid_placements.append(block_placements)
        # Update progress
        progress = (len(valid_placements) / len(blocks)) * 0.2  # 20% for preparation
        progress_bar.progress(progress)
        st.session_state.solving_progress = progress

    # Best solution tracking
    best_solution = None
    best_adjacency_count = -1

    # Backtracking function
    def backtrack(curr_grid, placed_idx, placed_info):
        nonlocal best_solution, best_adjacency_count

        # Check if we're out of time
        if time.time() - start_time > time_limit:
            return False

        # Update progress periodically
        progress = 0.2 + 0.8 * (len(placed_idx) / len(blocks))
        if int(progress * 100) != int(st.session_state.solving_progress * 100):
            progress_bar.progress(progress)
            st.session_state.solving_progress = progress
            status_text.text(
                f"Solving: Placed {len(placed_idx)}/{len(blocks)} blocks..."
            )

        # Check if all blocks are placed
        if len(placed_idx) == len(blocks):
            # Check if adjacency constraints are satisfied
            if check_adjacency_constraints_fast(curr_grid, adjacencies):
                # Count total adjacencies for optimization
                adj_count = count_adjacencies_fast(curr_grid, len(blocks))

                if adj_count > best_adjacency_count:
                    best_adjacency_count = adj_count
                    best_solution = (
                        copy.deepcopy(curr_grid),
                        placed_info.copy(),
                        adj_count,
                    )

                    # If not maximizing adjacencies, return the first valid solution
                    if not maximize_adjacencies:
                        return True

            # Continue searching if maximizing adjacencies
            return maximize_adjacencies

        # Next block to place
        remaining = [
            i for i in range(len(block_order)) if block_order[i] not in placed_idx
        ]

        # Heuristic: Try blocks with more constraints first
        def block_priority(block_idx):
            # Count how many adjacency constraints involve this block
            constraint_count = sum(1 for adj in adjacencies if block_idx in adj)
            return constraint_count

        # Sort remaining blocks by constraint priority
        remaining.sort(key=lambda i: block_priority(block_order[i]), reverse=True)

        for i in remaining:
            block_idx = block_order[i]

            # No valid placements for this block
            if not valid_placements[i]:
                continue

            for row, col, rot_idx, rot in valid_placements[i]:
                # Check if placement is still valid (might be occupied now)
                if can_place(curr_grid, rot, row, col):
                    # Place the block
                    place(curr_grid, rot, row, col, block_idx)
                    placed_idx.append(block_idx)
                    placed_info.append((block_idx, row, col, rot_idx, rot))

                    # Recurse
                    if backtrack(curr_grid, placed_idx, placed_info):
                        return True

                    # Backtrack
                    remove(curr_grid, rot, row, col)
                    placed_idx.pop()
                    placed_info.pop()

        return False

    # Start the backtracking
    grid_copy = copy.deepcopy(grid)
    backtrack(grid_copy, [], [])

    # Return the best solution found, if any
    if best_solution:
        return True, best_solution[1], best_solution[0], best_solution[2]
    else:
        return False, [], None, 0


# OPTIMIZATION: parallel solver for large problems
def parallel_solver_task(
    grid, blocks, adjacencies, block_ordering, maximize_adjacencies, time_slice=5
):
    """Task function for parallel solver"""
    # Convert grid to numpy if not already
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid, dtype=np.int8)

    # Start the backtracking with the given block ordering
    curr_grid = copy.deepcopy(grid)
    placed_idx = []
    placed_info = []
    best_solution = None
    best_adjacency_count = -1

    # Start time
    start_time = time.time()

    # Backtracking function
    def backtrack(curr_grid, placed_idx, placed_info):
        nonlocal best_solution, best_adjacency_count

        # Check if we're out of time
        if time.time() - start_time > time_slice:
            return False

        # Check if all blocks are placed
        if len(placed_idx) == len(blocks):
            # Check if adjacency constraints are satisfied
            if check_adjacency_constraints_fast(curr_grid, adjacencies):
                # Count total adjacencies for optimization
                adj_count = count_adjacencies_fast(curr_grid, len(blocks))

                if adj_count > best_adjacency_count:
                    best_adjacency_count = adj_count
                    best_solution = (
                        copy.deepcopy(curr_grid),
                        placed_info.copy(),
                        adj_count,
                    )

                    # If not maximizing adjacencies, return the first valid solution
                    if not maximize_adjacencies:
                        return True

            # Continue searching if maximizing adjacencies
            return maximize_adjacencies

        # Next block to place
        if placed_idx:
            next_block_idx = block_ordering[len(placed_idx)]
        else:
            next_block_idx = block_ordering[0]

        block = blocks[next_block_idx]
        rotations = get_rots(block)

        # Try all positions on the grid
        for rot_idx, rot in enumerate(rotations):
            for row in range(curr_grid.shape[0]):
                for col in range(curr_grid.shape[1]):
                    if can_place(curr_grid, rot, row, col):
                        # Place the block
                        place(curr_grid, rot, row, col, next_block_idx)
                        placed_idx.append(next_block_idx)
                        placed_info.append((next_block_idx, row, col, rot_idx, rot))

                        # Recurse
                        if backtrack(curr_grid, placed_idx, placed_info):
                            return True

                        # Backtrack
                        remove(curr_grid, rot, row, col)
                        placed_idx.pop()
                        placed_info.pop()

        return False

    # Start the backtracking
    backtrack(curr_grid, placed_idx, placed_info)

    # Return the best solution found, if any
    if best_solution:
        return True, best_solution[1], best_solution[0], best_solution[2]
    else:
        return False, [], None, 0


def solve_with_parallel_search(
    grid, blocks, adjacencies, maximize_adjacencies=False, time_limit=30
):
    """Solve using parallel search with different block orderings"""
    # Number of workers to use (adjust based on available cores)
    num_workers = min(multiprocessing.cpu_count(), 4)

    # Create different block orderings to try in parallel
    orderings = []

    # Calculate block areas
    block_areas = [block_cell_count(block) for block in blocks]

    # Order by size (largest first)
    size_order = list(
        sorted(range(len(blocks)), key=lambda i: block_areas[i], reverse=True)
    )
    orderings.append(size_order)

    # Order by constraint count
    def constraint_count(block_idx):
        return sum(1 for adj in adjacencies if block_idx in adj)

    constraint_order = list(
        sorted(range(len(blocks)), key=constraint_count, reverse=True)
    )
    orderings.append(constraint_order)

    # Add some random orderings
    for _ in range(num_workers - 2):
        order = list(range(len(blocks)))
        random.shuffle(order)
        orderings.append(order)

    # Fill remaining workers with shuffled versions of size and constraint orders
    while len(orderings) < num_workers:
        order = size_order.copy() if random.random() < 0.5 else constraint_order.copy()
        random.shuffle(order)
        orderings.append(order)

    # Progress display
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Starting parallel solvers...")

    # Run parallel tasks
    best_solution = None
    best_adjacency_count = -1

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        futures = []
        time_per_task = time_limit / num_workers

        for i, ordering in enumerate(orderings):
            futures.append(
                executor.submit(
                    parallel_solver_task,
                    grid,
                    blocks,
                    adjacencies,
                    ordering,
                    maximize_adjacencies,
                    time_per_task,
                )
            )

        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            success, placed_blocks, solution_grid, adj_count = future.result()

            # Update progress
            progress = (i + 1) / num_workers
            progress_bar.progress(progress)
            status_text.text(f"Processing solver results ({i+1}/{num_workers})...")

            if success and adj_count > best_adjacency_count:
                best_adjacency_count = adj_count
                best_solution = (success, placed_blocks, solution_grid, adj_count)

                # If not maximizing, we can stop once we find a solution
                if not maximize_adjacencies:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

    if best_solution:
        return best_solution
    else:
        return False, [], None, 0


# Smart solver that decides which method to use based on problem size
def smart_solve(grid, blocks, adjacencies, maximize_adjacencies=False):
    """Smart solver that chooses the appropriate algorithm based on problem size"""
    # Progress display
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Analyzing problem...")

    # Convert grid to numpy if not already
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid, dtype=np.int8)

    # Count cells in shape
    shape_size = np.sum(grid == 1)

    # Calculate total block area
    total_block_area = sum(block_cell_count(block) for block in blocks)

    # Choose solver based on problem characteristics
    block_count = len(blocks)

    if block_count <= 10:
        # For small problems, use backtracking with a generous time limit
        status_text.text("Using backtracking solver...")
        return solve_with_backtracking(
            grid, blocks, adjacencies, maximize_adjacencies, time_limit=30
        )
    elif block_count <= 20:
        # For medium problems, use parallel search with a moderate time limit
        status_text.text("Using parallel solver for medium problem...")
        return solve_with_parallel_search(
            grid, blocks, adjacencies, maximize_adjacencies, time_limit=45
        )
    else:
        # For large problems, use parallel search with a longer time limit
        status_text.text("Using parallel solver for large problem...")
        return solve_with_parallel_search(
            grid, blocks, adjacencies, maximize_adjacencies, time_limit=60
        )


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
    if st.button("Undo Last Point"):
        if st.session_state.points:
            removed = st.session_state.points.pop()
            st.success(f"Removed last point: {removed}")
            # Reset closing state and grid if undoing after closing
            st.session_state.is_closed = False
            st.session_state.grid = None
            st.session_state.area = None
            st.rerun()

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

        # Optional bulk block adding
        st.write("---")
        st.write("Bulk add multiple blocks at once:")
        bulk_text = st.text_area(
            "Enter blocks as 'length,width' (one per line)", placeholder="2,3\n1,2\n3,1"
        )

        if st.button("Add Bulk Blocks"):
            try:
                lines = bulk_text.strip().split("\n")
                added = 0
                for line in lines:
                    if line.strip():
                        length, width = map(int, line.strip().split(","))
                        if 1 <= length <= GRID_SIZE and 1 <= width <= GRID_SIZE:
                            block = [[1 for _ in range(width)] for _ in range(length)]
                            st.session_state.blocks.append(block)
                            added += 1
                if added > 0:
                    st.success(f"Added {added} blocks from bulk input")
            except Exception as e:
                st.error(f"Error adding bulk blocks: {str(e)}")

        # List of blocks
        if st.session_state.blocks:
            st.subheader("📦 Your Blocks")
            for i, block in enumerate(st.session_state.blocks):
                st.write(f"Block {i+1}: {len(block)} x {len(block[0])}")

            if st.button("Clear All Blocks"):
                st.session_state.blocks = []
                st.session_state.placed_blocks = []
                st.session_state.solution_grid = None
                st.session_state.solution_found = False
                st.success("All blocks cleared")
                st.rerun()

            # Delete specific block logic
            with st.expander("🗑️ Delete Block"):
                block_options = list(range(1, len(st.session_state.blocks) + 1))
                selected_block_to_delete = st.selectbox(
                    "Select block to delete",
                    options=block_options,
                    format_func=lambda x: f"Block {x}",
                    key="delete_block_select",
                )
                if st.button("Delete Selected Block"):
                    del st.session_state.blocks[selected_block_to_delete - 1]
                    st.success(f"Deleted Block {selected_block_to_delete}")
                    st.rerun()

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
                if st.session_state.grid is not None:
                    with st.spinner("Finding solution..."):
                        # Use our new smart solver
                        success, placed_blocks, solution_grid, adj_count = smart_solve(
                            st.session_state.grid,
                            st.session_state.blocks,
                            st.session_state.adjacencies,
                            maximize_adjacencies=maximize_button,
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
                            if isinstance(solution_grid, np.ndarray):
                                # Build adjacency map for fast checking
                                block_positions = build_adjacency_map(solution_grid)
                                req_adj_satisfied = sum(
                                    1
                                    for adj in st.session_state.adjacencies
                                    if are_adjacent_fast(
                                        block_positions, adj[0], adj[1]
                                    )
                                )
                            else:
                                # Fallback to standard checking
                                req_adj_satisfied = sum(
                                    1
                                    for adj in st.session_state.adjacencies
                                    if are_adjacent_fast(
                                        build_adjacency_map(np.array(solution_grid)),
                                        adj[0],
                                        adj[1],
                                    )
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
            st.session_state.solving_progress = 0
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
        if isinstance(grid, list):
            grid = np.array(grid)

        # Use a colormap with more distinct colors
        cmap = plt.cm.get_cmap("tab20", len(st.session_state.blocks))

        # Draw filled blocks
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] > 1:  # Block ID (+2)
                    block_id = grid[i, j] - 2
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
            # Build adjacency map once for efficiency
            block_positions = build_adjacency_map(grid)

            # Find all adjacencies
            highlighted_adjacencies = set()

            for block1_id in range(len(st.session_state.blocks)):
                for block2_id in range(block1_id + 1, len(st.session_state.blocks)):
                    # Check if these blocks are adjacent
                    grid_block1 = block1_id + 2
                    grid_block2 = block2_id + 2

                    # Skip if either block is not in the solution
                    if (
                        grid_block1 not in block_positions
                        or grid_block2 not in block_positions
                    ):
                        continue

                    # Find adjacent cells between the blocks
                    positions1 = block_positions[grid_block1]
                    positions2 = block_positions[grid_block2]

                    # Check for adjacency
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    adjacency_found = False

                    for i, j in positions1:
                        if adjacency_found:
                            break

                        for di, dj in directions:
                            ni, nj = i + di, j + dj
                            if (ni, nj) in positions2:
                                # Found adjacency, highlight it
                                is_required = (
                                    block1_id,
                                    block2_id,
                                ) in st.session_state.adjacencies or (
                                    block2_id,
                                    block1_id,
                                ) in st.session_state.adjacencies

                                # Draw a thicker line for required adjacencies
                                line_width = 4 if is_required else 2
                                line_style = "-" if is_required else "--"
                                line_color = "green" if is_required else "black"

                                # Draw line between cell centers
                                ax.plot(
                                    [j + 0.5, nj + 0.5],
                                    [GRID_SIZE - i + 0.5, GRID_SIZE - ni + 0.5],
                                    line_style,
                                    color=line_color,
                                    linewidth=line_width,
                                )

                                # Remember this adjacency to avoid duplicates
                                highlighted_adjacencies.add((block1_id, block2_id))
                                adjacency_found = True
                                break

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

            # Get the grid and create block positions map
            grid = st.session_state.solution_grid
            if isinstance(grid, list):
                grid = np.array(grid)

            block_positions = build_adjacency_map(grid)

            # Fill adjacency matrix
            for i in range(num_blocks):
                for j in range(i + 1, num_blocks):
                    if are_adjacent_fast(block_positions, i, j):
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

    # Performance metrics
    if st.session_state.solution_found:
        st.subheader("Performance Metrics")
        st.write(f"Total blocks placed: {len(st.session_state.blocks)}")
        st.write(f"Total adjacencies: {st.session_state.adjacency_count}")

        # Calculate efficiency metrics
        total_cells = sum(block_cell_count(block) for block in st.session_state.blocks)
        shape_area = st.session_state.area if st.session_state.area is not None else 0

        if shape_area > 0:
            efficiency = (total_cells / shape_area) * 100
            st.write(f"Shape fill efficiency: {efficiency:.2f}%")
