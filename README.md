

### ✅ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MrKrishnabhagat/updated.git
   cd Gplan
   ```

2. **Install Requirements**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the App**
   Use Streamlit to run the app:
   ```bash
   streamlit run gplan.py
   ```

4. **Start Building**
   - Draw shapes on a grid using click-and-drag.
   - Add rectangular blocks.
   - Define adjacency constraints.
   - Solve to optimally place blocks within the defined shape.

The core algorithms use backtracking with optimization techniques to efficiently place blocks. The solver prioritizes larger blocks and corner/edge positions first. It performs early constraint checking to prune invalid paths quickly and uses a two-phase approach that combines thorough searching with time-aware optimization. The system tracks block adjacencies, ensuring required blocks touch while maximizing total adjacencies when requested
