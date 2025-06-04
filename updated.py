import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import matplotlib.colors as mcolors
from matplotlib.path import Path
import copy
import random
import time
from collections import defaultdict

st.set_page_config(layout="wide")
st.title("🏠 Housing Floor Planner")

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
if "block_types" not in st.session_state:
    st.session_state.block_types = []
if "solution_grid" not in st.session_state:
    st.session_state.solution_grid = None
if "solution_found" not in st.session_state:
    st.session_state.solution_found = False
if "placed_blocks" not in st.session_state:
    st.session_state.placed_blocks = []
if "adjacencies" not in st.session_state:
    st.session_state.adjacencies = []
if "adjacency_count" not in st.session_state:
    st.session_state.adjacency_count = 0

# Grid settings
GRID_SIZE = 20

# Room type definitions with minimum dimensions and labels
ROOM_TYPES = {
    "Bedroom": {"min_length": 3, "min_width": 3, "color": "#FFB6C1", "shapes": ["rectangle"], "label": "Bd"},
    "Bathroom": {"min_length": 2, "min_width": 2, "color": "#87CEEB", "shapes": ["rectangle"], "label": "Bt"},
    "Living Room": {"min_length": 4, "min_width": 4, "color": "#98FB98", "shapes": ["rectangle", "L-shaped", "T-shaped"], "label": "LR"},
    "Kitchen": {"min_length": 3, "min_width": 2, "color": "#F0E68C", "shapes": ["rectangle"], "label": "K"}
}

# Shoelace area formula
def compute_area(points):
    if len(points) < 3:
        return 0
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    x.append(points[0][0])
    y.append(points[0][1])
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(len(points))))

# Create special shaped blocks
def create_l_shaped_block(length, width):
    """Create an L-shaped block with given dimensions"""
    if length < 3 or width < 3:
        return None
    
    block = [[1 for _ in range(width)] for _ in range(length)]
    corner_height = length // 2
    corner_width = width // 2
    
    for i in range(corner_height):
        for j in range(width - corner_width, width):
            block[i][j] = 0
    
    return block

def create_t_shaped_block(length, width):
    """Create a T-shaped block with given dimensions"""
    if length < 3 or width < 3:
        return None
    
    block = [[0 for _ in range(width)] for _ in range(length)]
    top_height = length // 3
    
    for i in range(top_height):
        for j in range(width):
            block[i][j] = 1
    
    stem_start = width // 3
    stem_end = 2 * width // 3
    for i in range(top_height, length):
        for j in range(stem_start, stem_end):
            block[i][j] = 1
    
    return block

def create_room_block(room_type, length, width, shape="rectangle"):
    """Create a block based on room type and shape"""
    if shape == "rectangle":
        return [[1 for _ in range(width)] for _ in range(length)]
    elif shape == "L-shaped":
        return create_l_shaped_block(length, width)
    elif shape == "T-shaped":
        return create_t_shaped_block(length, width)
    else:
        return [[1 for _ in range(width)] for _ in range(length)]

# Optimized rotation functions
def rotate_90_clockwise(block):
    """Rotate block 90 degrees clockwise"""
    return [list(row) for row in zip(*block[::-1])]

def get_rots_optimized(block):
    """Get all unique rotations of a block efficiently"""
    rotations = []
    current = block
    seen = set()
    
    for _ in range(4):
        # Convert to tuple for hashing
        block_tuple = tuple(tuple(row) for row in current)
        if block_tuple not in seen:
            seen.add(block_tuple)
            rotations.append([row[:] for row in current])  # Deep copy
        current = rotate_90_clockwise(current)
    
    return rotations

# Find the center point of a room in the grid
def find_room_center(grid, room_id):
    """Find the center point of a room for labeling"""
    room_cells = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == room_id + 2:
                room_cells.append((i, j))
    
    if not room_cells:
        return None
    
    avg_i = sum(cell[0] for cell in room_cells) / len(room_cells)
    avg_j = sum(cell[1] for cell in room_cells) / len(room_cells)
    
    center_i = round(avg_i)
    center_j = round(avg_j)
    
    if (center_i, center_j) in room_cells:
        return (center_i, center_j)
    else:
        min_dist = float('inf')
        closest_cell = room_cells[0]
        for cell in room_cells:
            dist = (cell[0] - avg_i)**2 + (cell[1] - avg_j)**2
            if dist < min_dist:
                min_dist = dist
                closest_cell = cell
        return closest_cell

# Create a grid representation of the shape
def create_grid_from_shape(points, grid_size):
    grid = [[0 for _ in range(grid_size + 1)] for _ in range(grid_size + 1)]

    if len(points) >= 3:
        path = Path(points)
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                if path.contains_point((j + 0.5, i + 0.5)):
                    grid[grid_size - i][j] = 1

    return grid

# Optimized placement functions
def can_place(grid, block, row, col):
    """Check if block can be placed at position with bounds checking"""
    block_height = len(block)
    block_width = len(block[0]) if block_height > 0 else 0
    
    # Quick bounds check
    if row + block_height > len(grid) or col + block_width > len(grid[0]):
        return False
    
    for i in range(block_height):
        for j in range(block_width):
            if block[i][j] == 1 and grid[row + i][col + j] != 1:
                return False
    return True

def place(grid, block, row, col, block_id):
    for i in range(len(block)):
        for j in range(len(block[i])):
            if block[i][j] == 1:
                grid[row + i][col + j] = block_id + 2

def remove(grid, block, row, col):
    for i in range(len(block)):
        for j in range(len(block[i])):
            if block[i][j] == 1:
                grid[row + i][col + j] = 1

def are_adjacent(grid, block1_id, block2_id):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == block1_id + 2:
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and 
                        grid[ni][nj] == block2_id + 2):
                        return True
    return False

def count_adjacencies(grid, block_count):
    adjacency_count = 0
    for block1_id in range(block_count):
        for block2_id in range(block1_id + 1, block_count):
            if are_adjacent(grid, block1_id, block2_id):
                adjacency_count += 1
    return adjacency_count

def check_adjacency_constraints(grid, adjacencies):
    for block1_id, block2_id in adjacencies:
        if not are_adjacent(grid, block1_id, block2_id):
            return False
    return True

def block_cell_count(block):
    return sum(sum(row) for row in block)

# Optimized solving algorithm
def solve_with_optimized_algorithm(grid, blocks, adjacencies, time_limit=15):
    """Optimized solver with better performance"""
    start_time = time.time()
    best_solution = None
    best_adj_count = -1
    
    # Pre-compute rotations for all blocks
    block_rotations = [get_rots_optimized(block) for block in blocks]
    
    # Prioritize larger blocks first
    block_order = sorted(range(len(blocks)), 
                        key=lambda i: block_cell_count(blocks[i]), 
                        reverse=True)
    
    # Get valid positions more efficiently
    valid_positions = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                valid_positions.append((i, j))
    
    def backtrack_optimized(block_idx, current_grid, placed_blocks):
        nonlocal best_solution, best_adj_count
        
        if time.time() - start_time > time_limit:
            return False
        
        if block_idx >= len(block_order):
            if check_adjacency_constraints(current_grid, adjacencies):
                adj_count = count_adjacencies(current_grid, len(blocks))
                if adj_count > best_adj_count:
                    best_adj_count = adj_count
                    best_solution = (copy.deepcopy(current_grid), placed_blocks[:], adj_count)
                return True
            return False
        
        actual_block_id = block_order[block_idx]
        rotations = block_rotations[actual_block_id]
        
        # Try positions in a more strategic order
        for row, col in valid_positions:
            for rot_idx, rotation in enumerate(rotations):
                if can_place(current_grid, rotation, row, col):
                    place(current_grid, rotation, row, col, actual_block_id)
                    placed_blocks.append((actual_block_id, row, col, rot_idx, rotation))
                    
                    if backtrack_optimized(block_idx + 1, current_grid, placed_blocks):
                        if not adjacencies:  # If maximizing, continue searching
                            pass
                        else:  # If satisfying constraints, can return early
                            return True
                    
                    remove(current_grid, rotation, row, col)
                    placed_blocks.pop()
        
        return False
    
    # Try multiple attempts with different strategies
    max_attempts = 5 if adjacencies else 10
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        grid_copy = copy.deepcopy(grid)
        placed_blocks = []
        
        # Randomize position order for variety
        if attempt > 0:
            random.shuffle(valid_positions)
        
        backtrack_optimized(0, grid_copy, placed_blocks)
    
    if best_solution:
        return True, best_solution[1], best_solution[0], best_solution[2]
    else:
        return False, [], None, 0

# Room removal function
def remove_room(room_index):
    """Remove a room and update adjacencies"""
    if 0 <= room_index < len(st.session_state.blocks):
        # Remove the room
        st.session_state.blocks.pop(room_index)
        st.session_state.block_types.pop(room_index)
        
        # Update adjacencies - remove any adjacency involving this room
        # and adjust indices for rooms that come after
        new_adjacencies = []
        for adj in st.session_state.adjacencies:
            room1, room2 = adj
            if room1 != room_index and room2 != room_index:
                # Adjust indices for rooms after the removed room
                new_room1 = room1 if room1 < room_index else room1 - 1
                new_room2 = room2 if room2 < room_index else room2 - 1
                new_adjacencies.append((min(new_room1, new_room2), max(new_room1, new_room2)))
        
        st.session_state.adjacencies = new_adjacencies
        
        # Clear solution
        st.session_state.solution_found = False
        st.session_state.solution_grid = None
        st.session_state.placed_blocks = []
        st.session_state.adjacency_count = 0

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Draw Your House Shape")
    st.write("Enter coordinates to draw the house outline (x, y):")

    # Point input
    x = st.number_input(
        "X coordinate", min_value=0, max_value=GRID_SIZE, step=1, key="x_coord"
    )
    y = st.number_input(
        "Y coordinate", min_value=0, max_value=GRID_SIZE, step=1, key="y_coord"
    )

    # Add point button
    if st.button("Add Point"):
        if [x, y] not in st.session_state.points:
            if len(st.session_state.points) > 0:
                last_x, last_y = st.session_state.points[-1]
                if (x == last_x or y == last_y) and (x != last_x or y != last_y):
                    st.session_state.points.append([x, y])
                else:
                    st.warning("Only horizontal or vertical lines are allowed.")
            else:
                st.session_state.points.append([x, y])

    # Check if shape can be closed
    if len(st.session_state.points) >= 3:
        first_x, first_y = st.session_state.points[0]
        last_x, last_y = st.session_state.points[-1]

        can_close = (first_x == last_x or first_y == last_y) and (
            first_x != last_x or first_y != last_y
        )

        if not st.session_state.is_closed:
            close_button = st.button("Close Shape", disabled=not can_close)
            if close_button and can_close:
                st.session_state.points.append([first_x, first_y])
                st.session_state.is_closed = True
                st.session_state.area = compute_area(st.session_state.points[:-1])
                st.session_state.grid = create_grid_from_shape(st.session_state.points, GRID_SIZE)

    # Show points as a table
    if st.session_state.points:
        st.subheader("📍 House Outline Points")
        df = pd.DataFrame(st.session_state.points, columns=["X", "Y"])
        df.index = np.arange(1, len(df) + 1)
        st.dataframe(df)

    # Display area if shape is closed
    if st.session_state.is_closed and st.session_state.area is not None:
        st.success(f"✅ House outline complete — Area: **{st.session_state.area:.2f} square units**")

    # Room creation section
    if st.session_state.is_closed:
        st.subheader("2. Add Rooms to Your House")

        # Room type selection
        room_type = st.selectbox(
            "Select Room Type",
            options=list(ROOM_TYPES.keys()),
            key="room_type_select"
        )

        # Get minimum dimensions for selected room type
        min_length = ROOM_TYPES[room_type]["min_length"]
        min_width = ROOM_TYPES[room_type]["min_width"]
        available_shapes = ROOM_TYPES[room_type]["shapes"]

        # Shape selection (only for living room)
        if len(available_shapes) > 1:
            room_shape = st.selectbox(
                "Select Room Shape",
                options=available_shapes,
                key="room_shape_select"
            )
        else:
            room_shape = available_shapes[0]

        # Dimension inputs with minimum constraints
        col_len, col_width = st.columns(2)
        with col_len:
            room_length = st.number_input(
                f"Length (min: {min_length})", 
                min_value=min_length, 
                max_value=GRID_SIZE, 
                value=min_length
            )
        with col_width:
            room_width = st.number_input(
                f"Width (min: {min_width})", 
                min_value=min_width, 
                max_value=GRID_SIZE, 
                value=min_width
            )

        # Display room info
        st.info(f"**{room_type}** - Shape: {room_shape} - Dimensions: {room_length} x {room_width}")

        if st.button("Add Room"):
            # Create the room block
            room_block = create_room_block(room_type, room_length, room_width, room_shape)
            
            if room_block is not None:
                st.session_state.blocks.append(room_block)
                st.session_state.block_types.append({
                    "type": room_type,
                    "shape": room_shape,
                    "dimensions": (room_length, room_width)
                })
                st.success(f"Added {room_type} ({room_shape}) with dimensions {room_length} x {room_width}")
                
                # Clear solution when adding new room
                st.session_state.solution_found = False
                st.session_state.solution_grid = None
                st.session_state.placed_blocks = []
                st.session_state.adjacency_count = 0
            else:
                st.error("Could not create room with specified dimensions and shape")

        # List of rooms with remove buttons
        if st.session_state.blocks:
            st.subheader("🏠 Your Rooms")
            for i, (block, room_info) in enumerate(zip(st.session_state.blocks, st.session_state.block_types)):
                room_type = room_info["type"]
                room_shape = room_info["shape"]
                dimensions = room_info["dimensions"]
                
                col_room, col_remove = st.columns([3, 1])
                with col_room:
                    st.write(f"**Room {i+1}**: {room_type} ({room_shape}) - {dimensions[0]} x {dimensions[1]}")
                with col_remove:
                    if st.button("🗑️", key=f"remove_room_{i}", help=f"Remove {room_type} {i+1}"):
                        remove_room(i)
                        st.rerun()

        # Adjacency constraints section
        if len(st.session_state.blocks) >= 2:
            st.subheader("3. Define Room Adjacencies (Optional)")
            st.write("⚡ **Auto-maximize**: If no adjacencies are specified, the system will automatically maximize room connections!")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                room1_idx = st.selectbox(
                    "Room 1",
                    options=list(range(len(st.session_state.blocks))),
                    format_func=lambda x: f"{st.session_state.block_types[x]['type']} {x+1}",
                    key="room1_select"
                )
            with col_r2:
                room2_idx = st.selectbox(
                    "Room 2",
                    options=list(range(len(st.session_state.blocks))),
                    format_func=lambda x: f"{st.session_state.block_types[x]['type']} {x+1}",
                    key="room2_select"
                )

            if st.button("Add Room Adjacency"):
                if room1_idx != room2_idx:
                    adj = (min(room1_idx, room2_idx), max(room1_idx, room2_idx))
                    
                    if adj not in st.session_state.adjacencies:
                        st.session_state.adjacencies.append(adj)
                        room1_name = f"{st.session_state.block_types[room1_idx]['type']} {room1_idx+1}"
                        room2_name = f"{st.session_state.block_types[room2_idx]['type']} {room2_idx+1}"
                        st.success(f"Added adjacency between {room1_name} and {room2_name}")
                    else:
                        st.warning("This adjacency already exists!")
                else:
                    st.warning("Cannot add adjacency between the same room!")

            # Show adjacencies list with remove buttons
            if st.session_state.adjacencies:
                st.subheader("Required Room Adjacencies:")
                for idx, (r1, r2) in enumerate(st.session_state.adjacencies):
                    room1_name = f"{st.session_state.block_types[r1]['type']} {r1+1}"
                    room2_name = f"{st.session_state.block_types[r2]['type']} {r2+1}"
                    
                    col_adj, col_remove_adj = st.columns([4, 1])
                    with col_adj:
                        st.write(f"• {room1_name} ↔ {room2_name}")
                    with col_remove_adj:
                        if st.button("❌", key=f"remove_adj_{idx}", help="Remove this adjacency"):
                            st.session_state.adjacencies.pop(idx)
                            st.rerun()

                if st.button("Clear All Adjacencies"):
                    st.session_state.adjacencies = []
                    st.success("All adjacencies cleared")

        # Solve button
        if st.session_state.blocks:
            st.subheader("4. Generate Floor Plan")

            # Calculate total area of rooms
            total_cells = sum(block_cell_count(block) for block in st.session_state.blocks)
            house_area = st.session_state.area if st.session_state.area is not None else 0

            # Display room area info
            st.info(f"Total room area: {total_cells} square units")
            if total_cells > house_area:
                st.warning(f"Warning: Total room area ({total_cells}) exceeds house area ({house_area:.2f})")

            # Single solve button that handles both cases
            if st.button("🚀 Generate Optimized Floor Plan"):
                if st.session_state.grid:
                    # Determine if we should maximize adjacencies
                    maximize_adj = len(st.session_state.adjacencies) == 0
                    
                    if maximize_adj:
                        st.info("🔄 No specific adjacencies defined - automatically maximizing room connections...")
                    else:
                        st.info(f"🔄 Generating layout with {len(st.session_state.adjacencies)} required adjacencies...")
                    
                    with st.spinner("Generating optimized floor plan..."):
                        success, placed_blocks, solution_grid, adj_count = solve_with_optimized_algorithm(
                            st.session_state.grid,
                            st.session_state.blocks,
                            st.session_state.adjacencies,
                            time_limit=15
                        )

                        st.session_state.solution_found = success
                        st.session_state.placed_blocks = placed_blocks
                        st.session_state.adjacency_count = adj_count

                        if success:
                            st.session_state.solution_grid = solution_grid
                            st.success(f"✅ Floor plan generated! All {len(st.session_state.blocks)} rooms placed successfully.")
                            st.info(f"🔗 Total room connections: {adj_count}")

                            if st.session_state.adjacencies:
                                req_adj_satisfied = sum(
                                    1 for adj in st.session_state.adjacencies
                                    if are_adjacent(solution_grid, adj[0], adj[1])
                                )
                                st.success(f"Required adjacencies satisfied: {req_adj_satisfied}/{len(st.session_state.adjacencies)}")
                            else:
                                st.success("🎯 Automatically maximized room connections for optimal layout!")
                        else:
                            st.error("❌ Could not generate floor plan. Try adjusting room sizes or adjacency requirements.")

        # Reset button
        if st.button("Reset Everything"):
            for key in ["points", "grid", "area", "is_closed", "blocks", "block_types", 
                       "solution_grid", "solution_found", "placed_blocks", "adjacencies", "adjacency_count"]:
                if key in st.session_state:
                    if key in ["points", "blocks", "block_types", "placed_blocks", "adjacencies"]:
                        st.session_state[key] = []
                    elif key in ["solution_found", "is_closed"]:
                        st.session_state[key] = False
                    elif key == "adjacency_count":
                        st.session_state[key] = 0
                    else:
                        st.session_state[key] = None
            st.rerun()

# Visualization column (same as before but with improved performance)
with col2:
    st.subheader("Floor Plan Visualization")

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 12))

    # Set up the plot
    ax.set_xlim(-0.5, GRID_SIZE + 0.5)
    ax.set_ylim(-0.5, GRID_SIZE + 0.5)
    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)

    # Draw the house outline
    if len(st.session_state.points) > 0:
        points_array = np.array(st.session_state.points)
        ax.plot(points_array[:, 0], points_array[:, 1], "b-", linewidth=3, label="House Outline")

        for i, (x, y) in enumerate(st.session_state.points):
            ax.plot(x, y, "bo", markersize=6)
            ax.text(x + 0.2, y + 0.2, f"({x},{y})", fontsize=8)

        if st.session_state.is_closed and len(st.session_state.points) > 3:
            polygon = Polygon(points_array[:-1], alpha=0.1, color="lightblue", label="House Area")
            ax.add_patch(polygon)

    # Show solution if available
    if st.session_state.solution_found and st.session_state.solution_grid is not None:
        grid = st.session_state.solution_grid

        # Draw filled rooms with appropriate colors
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] > 1:  # Room ID (+2)
                    room_id = grid[i][j] - 2
                    room_info = st.session_state.block_types[room_id]
                    room_type = room_info["type"]
                    color = ROOM_TYPES[room_type]["color"]
                    
                    ax.add_patch(
                        Rectangle((j, GRID_SIZE - i), 1, 1, color=color, alpha=0.8, 
                                edgecolor='black', linewidth=0.5)
                    )

        # Add room labels only at center points
        for room_id in range(len(st.session_state.blocks)):
            center = find_room_center(grid, room_id)
            if center:
                center_i, center_j = center
                room_info = st.session_state.block_types[room_id]
                room_type = room_info["type"]
                label = ROOM_TYPES[room_type]["label"]
                
                # Add room label at center
                ax.text(
                    center_j + 0.5, GRID_SIZE - center_i + 0.5,
                    f"{label}{room_id + 1}",
                    ha="center", va="center",
                    color="black", fontweight="bold", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8)
                )

        # Highlight adjacencies
        if st.checkbox("Show Room Connections", value=True):
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if grid[i][j] > 1:
                        room_id = grid[i][j] - 2
                        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                        for di, dj in directions:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and 
                                grid[ni][nj] > 1 and grid[ni][nj] != grid[i][j]):
                                
                                neighbor_id = grid[ni][nj] - 2
                                if room_id < neighbor_id:
                                    is_required = (room_id, neighbor_id) in st.session_state.adjacencies
                                    line_width = 4 if is_required else 2
                                    line_color = "red" if is_required else "gray"
                                    alpha = 0.8 if is_required else 0.4

                                    ax.plot(
                                        [j + 0.5, nj + 0.5],
                                        [GRID_SIZE - i + 0.5, GRID_SIZE - ni + 0.5],
                                        "-", color=line_color, linewidth=line_width, alpha=alpha
                                    )

    # Add legend for room types
    if st.session_state.blocks:
        legend_elements = []
        for room_type, properties in ROOM_TYPES.items():
            # Check if this room type is used
            if any(block_info["type"] == room_type for block_info in st.session_state.block_types):
                legend_elements.append(
                    Rectangle((0, 0), 1, 1, facecolor=properties["color"], 
                            alpha=0.8, label=f"{room_type} ({properties['label']})")
                )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    ax.set_title("House Floor Plan", fontsize=16, fontweight="bold")
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)

    # Display the plot
    st.pyplot(fig)

    # Display room summary
    if st.session_state.blocks:
        st.subheader("🏠 Room Summary")
        room_summary = {}
        for room_info in st.session_state.block_types:
            room_type = room_info["type"]
            if room_type in room_summary:
                room_summary[room_type] += 1
            else:
                room_summary[room_type] = 1
        
        for room_type, count in room_summary.items():
            label = ROOM_TYPES[room_type]["label"]
            st.write(f"**{room_type} ({label})**: {count} room(s)")

    # Display placement details if solution found
    if st.session_state.solution_found and st.session_state.placed_blocks:
        if st.checkbox("Show Room Placement Details"):
            st.subheader("Room Placement Details")
            for room_id, row, col, rot_idx, _ in st.session_state.placed_blocks:
                room_info = st.session_state.block_types[room_id]
                room_type = room_info["type"]
                room_shape = room_info["shape"]
                dimensions = room_info["dimensions"]
                label = ROOM_TYPES[room_type]["label"]
                
                st.write(f"**{room_type} {room_id+1} ({label}{room_id+1})** ({room_shape}) - "
                        f"Size: {dimensions[0]}x{dimensions[1]} - "
                        f"Position: ({col},{GRID_SIZE-row}) - "
                        f"Rotation: {rot_idx*90}°")

    # Display adjacency matrix if solution found
    if st.session_state.solution_found and st.session_state.placed_blocks:
        if st.checkbox("Show Room Adjacency Matrix"):
            st.subheader("Room Adjacency Matrix")

            num_rooms = len(st.session_state.blocks)
            adj_matrix = [[0 for _ in range(num_rooms)] for _ in range(num_rooms)]

            for i in range(num_rooms):
                for j in range(i + 1, num_rooms):
                    if are_adjacent(st.session_state.solution_grid, i, j):
                        adj_matrix[i][j] = adj_matrix[j][i] = 1

            room_labels = [f"{st.session_state.block_types[i]['type']} {i+1}" 
                          for i in range(num_rooms)]
            
            adj_df = pd.DataFrame(adj_matrix, index=room_labels, columns=room_labels)
            st.dataframe(adj_df)
