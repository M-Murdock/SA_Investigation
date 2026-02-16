# Dot Mover Simulator - JavaScript Conversion (sim.py)

This is a JavaScript conversion of the refactored Python pygame-based dot simulator (`sim.py`) with shared autonomy features.

## Overview

The simulator demonstrates shared autonomy through a simple dot-moving task where:
- The user controls a dot using arrow keys
- An AI assistant infers the user's goal from multiple possible policies
- The final action is a blend of user input and AI assistance
- Three inference algorithms available: Bayesian, MaxEnt, and CRF
- Three arbitration strategies: Linear, Probabilistic, and User-Only

## Files

### Core Files
- **sim.html** - Main HTML page with canvas and UI controls
- **sim.js** - Complete JavaScript implementation
- **q_table_*.json** - Converted Q-learning policy tables (3 policies)
- **background_images/** - Background images for kitchen theme
  - `sugar.png` - Top-left corner
  - `egg.png` - Top-left corner
  - `cheese.png` - Top-right corner
  - `milk_carton.png` - Bottom-right corner
  - `mixing_bowl.png` - Center (large)
  - `spoon.png` - Cursor/agent image

### Original Python Files
- **sim.py** - Original refactored Python implementation

## How to Run

### Option 1: Simple Local Server (Recommended)
```bash
# Using Python
python3 -m http.server 8000

# Using Node.js
npx http-server

# Using PHP
php -S localhost:8000
```

Then open: `http://localhost:8000/sim.html`

### Option 2: Direct File Access
Open `sim.html` directly in your browser. The simulator will use synthetic Q-tables if the JSON files can't be loaded.

**Note on Background Images**: For background images to load, you'll need to run a local server. The images are configured to show:
- Kitchen ingredients (sugar, egg, cheese, milk) in the corners
- A mixing bowl in the center
- A spoon as the cursor/agent image

## Features

### Inference Types
1. **Bayesian** (Default)
   - Bayesian inference with exponential forgetting
   - Goal-switch prior for intent persistence
   - Posterior temperature smoothing

2. **Max Entropy**
   - Maximum entropy inverse optimal control
   - Exponentially weighted belief updates

3. **CRF (Conditional Random Fields)**
   - Linear-chain CRF with temporal smoothness
   - Pairwise potentials encourage action persistence

### Arbitration Types
1. **Linear**
   - Fixed blend: `action = γ * user + (1-γ) * robot`
   - Gamma controlled by slider (default: 0.4)

2. **Probabilistic** (Default)
   - Dynamic blending based on robot confidence
   - `action = confidence * robot + (1-confidence) * user`
   - Confidence = max probability across policies

3. **User Action Only**
   - No robot assistance
   - Direct user control for comparison

### Controls
- **Arrow Keys** - Move the dot (Up/Down/Left/Right)
- **Inference Type Dropdown** - Select goal inference algorithm
- **Arbitration Type Dropdown** - Select blending strategy
- **Gamma Slider** - Adjust Linear arbitration weight
- **Reset Button** - Reset simulation to center

## Architecture

### Main Classes

#### DotPolicy
```javascript
class DotPolicy {
    constructor(qTableFile, qTable)
    getQValue(state, action)
    getAction(state)
}
```
Wrapper for Q-learning policy tables. Stores state-action values in 3D array.

#### BayesianPredictor
```javascript
class BayesianPredictor {
    constructor(policies, actionSpaceSize, prior, tau, eps)
    logLikelihood(state, userAction, policy)
    update(state, userAction, alpha, pSwitch, beta)
    getProb()
}
```
Bayesian goal inference with:
- Softmax likelihood model
- Exponential forgetting (alpha)
- Goal-switch prior (pSwitch)
- Posterior smoothing (beta)

#### MaxEntPredictor
```javascript
class MaxEntPredictor {
    constructor(policies, actionSpaceSize, tau, eps)
    logLikelihood(state, userAction, policy)
    update(state, userAction, alpha)
    getProb()
}
```
Maximum Entropy IOC predictor with exponentially weighted updates.

#### CRFPredictor
```javascript
class CRFPredictor {
    constructor(policies, actionSpaceSize, eps, tau, pairwiseWeight, alpha, pSwitch, beta)
    logLikelihood(state, userAction, policy)
    update(state, userAction)
    unaryFn(policy, state, action)
    pairwiseFn(prevA, a)
    getProb()
}
```
Conditional Random Field with temporal smoothness.

#### SharedAutoPolicy
```javascript
class SharedAutoPolicy {
    constructor(policies, actionSpace)
    normalizeQValue(qValue, policyIdx)
    getAction(state, probPolicy, returnDist, sample)
}
```
Assistance provider that:
- Normalizes Q-values across policies
- Computes belief-weighted expected Q-values
- Returns optimal action

#### DotSimulator
```javascript
class DotSimulator {
    constructor(policyDir, inferenceType, assistanceType, arbitrationType)
    getState(x, y)
    indexToTuple(index)
    executeAction(action)
    ensureWithinBoundaries()
    blend(u, a)
    createPredictor()
    createAssistant()
    runShared()
}
```
Main simulation controller managing the game loop, rendering, and coordination.

## Technical Details

### State Space
- **Grid**: 30×30 discrete states
- **Mapping**: Continuous position → `(x÷20, y÷20)`
- **Boundaries**: 600×600 pixel canvas

### Action Space
- **4 discrete actions**: Up (0), Down (1), Left (2), Right (3)
- **Vectors**: Unit directions scaled by DOT_SPEED (5 pixels)

### Q-Table Format
```javascript
qTable[state_x][state_y][action]
// state_x, state_y: 0-29 (grid coordinates)
// action: 0-3 (up/down/left/right)
// value: float (Q-value for state-action pair)
```

### Blending Algorithm
1. **Get user action** u (0-3 or -1 for none)
2. **Inference**: Update belief p(policy|history)
3. **Assistance**: Compute optimal robot action a
4. **Arbitration**: Blend u and a based on strategy
5. **Normalize**: Ensure unit magnitude
6. **Execute**: Move dot by blended direction

### Performance
- **Frame Rate**: 60 FPS (requestAnimationFrame)
- **Real-time**: Inference runs every user action
- **Efficient**: Pre-computed Q-table normalization

## Key Differences from Python Version

### Similarities
- ✅ Identical algorithmic logic for all predictors
- ✅ Same Q-table structure (30×30×4)
- ✅ Same blending/arbitration strategies
- ✅ Same action and state spaces
- ✅ Same policy goals (top-left, bottom-right, orbit)

### Changes
- **Canvas API** instead of Pygame
- **requestAnimationFrame** instead of pygame clock
- **JSON Q-tables** instead of numpy .npy files
- **Event-based keyboard** instead of polling
- **HTML/CSS UI** instead of pygame GUI
- **Pure JavaScript** (no external dependencies)

### Simplifications
- No top panel with probability bars (simpler UI)
- No goal visualization overlays
- No checkboxes for toggling visualizations
- Probability display in text instead of bars
- Background images enabled and configured

## Customization

### Add New Policies
1. Create Q-table as 30×30×4 JSON array
2. Add to `loadPolicies()` function
3. Update policy names/descriptions

### Change Parameters
```javascript
// In DEFAULTS object
GAMMA: 0.4,          // Linear blend weight
DOT_SPEED: 5,        // Movement speed
GRID_SIZE: 20,       // State discretization
DOT_RADIUS: 8,       // Visual dot size

// Predictor hyperparameters
tau: 0.8,            // Softmax temperature
alpha: 0.05,         // Forgetting rate
pSwitch: 0.02,       // Goal-switch prior
eps: 0.001,          // Smoothing factor
```

### Synthetic Q-Tables
The code includes generators for demo policies:
- `createTopLeftPolicy()` - Move to (0,0)
- `createBottomRightPolicy()` - Move to (29,29)
- `createOrbitPolicy()` - Circle around center

These are used as fallback if JSON files aren't available.

## Understanding the Algorithms

### Bayesian Update
```
posterior ∝ likelihood × prior
P(policy|action) ∝ P(action|policy) × P(policy)
```
With temporal decay to prevent over-confidence.

### Shared Autonomy
```
Q_expected = Σ P(policy_i) × Q_i(state, action)
action* = argmax Q_expected
```
Weighted by belief distribution.

### Arbitration
- **Linear**: `(1-γ)·robot + γ·user`
- **Probabilistic**: `conf·robot + (1-conf)·user`
- **User**: `1·user + 0·robot`

All normalized to unit magnitude.

## Browser Compatibility

Tested and working in:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

Requires:
- ES6 support (arrow functions, classes, async/await)
- Canvas API
- Fetch API (for loading JSON Q-tables)

## Credits

Converted from Python pygame implementation (`sim.py`).
Preserves all core shared autonomy algorithms and behavior.

## License

Same as original Python implementation.
