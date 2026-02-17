// Dot Simulator in JavaScript
// Converted from Python pygame implementation (sim.py - refactored version)
// Behavior preserved from original Python implementation

// -------------------------
// Constants & Configuration
// -------------------------
const DEFAULTS = {
    GAMMA: 0.4,
    SCREEN_WIDTH: 600,
    SCREEN_HEIGHT: 600,
    CHECKBOX_SIZE: 20,
    DOT_RADIUS: 8,
    DOT_SPEED: 5,
    GRID_SIZE: 20,
    TEXT_SIZE: 15,
    CLICK_COOLDOWN: 50,
};

// Action index -> (dx, dy)
const INDEX_TO_TUPLE = {
    0: [0, -1],  // UP
    1: [0, 1],   // DOWN
    2: [-1, 0],  // LEFT
    3: [1, 0],   // RIGHT
};

// Enum-like objects
const Inference = {
    BAYESIAN: 'bayesian',
    MAX_ENT: 'maxent',
    CRF: 'crf'
};

const Assistance = {
    DISTRIBUTION: 'distribution'
};

const Arbitration = {
    LINEAR: 'linear',
    PROBABILISTIC: 'probabilistic',
    ONLY_USER: 'onlyuser'
};

// -------------------------
// Utility Functions
// -------------------------
function generateColors(n) {
    /**
     * Generate n colors that avoid pure 0 or 255 values (keeps distinct, non-black/white).
     */
    const colors = [];
    for (let i = 0; i < n; i++) {
        const r = ((i + 1) * 123) % 254;
        const g = ((i + 1) * 231) % 254;
        const b = ((i + 1) * 77) % 254;
        colors.push([r, g, b]);
    }
    return colors;
}

// -------------------------
// Lightweight Q-table wrapper
// -------------------------
class DotPolicy {
    /**
     * Simple adapter around a saved Q-table .npy file.
     * 
     * Public API:
     *   get_q_value(state, action)
     *   get_action(state) -> argmax over actions
     */
    constructor(qTableFile, qTable) {
        this.qTableFile = qTableFile;
        this.qTable = qTable; // 3D array: [state_x][state_y][action]
        console.log(`Loaded Q-table from ${qTableFile}`);
    }

    getQValue(state, action) {
        return this.qTable[state[0]][state[1]][action];
    }

    getAction(state) {
        const qValues = this.qTable[state[0]][state[1]];
        return qValues.indexOf(Math.max(...qValues));
    }
}

// -------------------------
// Bayesian Predictor
// -------------------------
class BayesianPredictor {
    constructor(policies, actionSpaceSize = 4, prior = null, tau = 0.8, eps = 1e-3) {
        this.policies = policies;
        this.N = policies.length;
        this.actionSpaceSize = actionSpaceSize;
        this.eps = eps;
        this.tau = tau;
        
        // Log posterior
        if (prior === null) {
            prior = new Array(this.N).fill(1.0 / this.N);
        }
        this.logPost = prior.map(p => Math.log(p + 1e-12));
    }

    logLikelihood(state, userAction, policy) {
        /**
         * Return log P(u | pi).
         */
        const Q = [];
        for (let a = 0; a < this.actionSpaceSize; a++) {
            Q.push(policy.getQValue(state, a));
        }

        // Softmax likelihood P(u | pi)
        const logits = Q.map(q => q / this.tau);
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(l => Math.exp(l - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(e => e / sumExp);

        return Math.log(probs[userAction] + 1e-8);
    }

    update(state, userAction, alpha = 0.05, pSwitch = 0.02, beta = 1.0) {
        /**
         * Bayesian update with forgetting, goal persistence, and posterior smoothing.
         */
        // 1. Compute log-likelihoods
        const logLikes = [];
        for (let i = 0; i < this.N; i++) {
            logLikes.push(this.logLikelihood(state, userAction, this.policies[i]));
        }

        // 2. Exponential forgetting (key fix)
        this.logPost = this.logPost.map((lp, i) => (1 - alpha) * lp + alpha * logLikes[i]);

        // 3. Normalize in probability space
        const maxLogP = Math.max(...this.logPost);
        let post = this.logPost.map(lp => Math.exp(lp - maxLogP));
        let sumPost = post.reduce((a, b) => a + b, 0);
        post = post.map(p => p / sumPost);

        // 4. Goal-switch prior (intent persistence)
        if (pSwitch > 0) {
            post = post.map(p => (1 - pSwitch) * p + pSwitch * (1.0 / this.N));
        }

        // 5. Posterior temperature (optional but useful)
        if (beta !== 1.0) {
            post = post.map(p => Math.pow(p, 1.0 / beta));
            sumPost = post.reduce((a, b) => a + b, 0);
            post = post.map(p => p / sumPost);
        }

        // 6. Light smoothing (numerical safety only)
        post = post.map(p => (1 - this.eps) * p + this.eps * (1.0 / this.N));

        // 7. Store back in log space
        this.logPost = post.map(p => Math.log(p + 1e-12));

        return post;
    }

    getProb() {
        const maxLogP = Math.max(...this.logPost);
        let post = this.logPost.map(lp => Math.exp(lp - maxLogP));
        const sumPost = post.reduce((a, b) => a + b, 0);
        return post.map(p => p / sumPost);
    }
}

// -------------------------
// MaxEnt Predictor
// -------------------------
class MaxEntPredictor {
    constructor(policies, actionSpaceSize = 4, tau = 0.8, eps = 1e-2) {
        this.policies = policies;
        this.N = policies.length;
        this.actionSpaceSize = actionSpaceSize;
        this.tau = tau;
        this.eps = eps;
        
        // Initialize uniform belief over policies
        this.logPost = new Array(this.N).fill(Math.log(1.0 / this.N + 1e-12));
    }

    logLikelihood(state, userAction, policy) {
        /**
         * MaxEnt IOC likelihood: P(u | pi) proportional to exp(Q(s,u)/tau).
         */
        const Q = [];
        for (let a = 0; a < this.actionSpaceSize; a++) {
            Q.push(policy.getQValue(state, a));
        }

        const logits = Q.map(q => q / this.tau);
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(l => Math.exp(l - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(e => e / sumExp);

        return Math.log(probs[userAction] + 1e-12);
    }

    update(state, userAction, alpha = 0.5) {
        /**
         * Update the belief over policies using MaxEnt likelihood.
         */
        const logLikes = [];
        for (let i = 0; i < this.N; i++) {
            logLikes.push(this.logLikelihood(state, userAction, this.policies[i]));
        }

        // Convert to probability
        const maxLogLike = Math.max(...logLikes);
        let likes = logLikes.map(ll => Math.exp(ll - maxLogLike));
        const sumLikes = likes.reduce((a, b) => a + b, 0);
        likes = likes.map(l => l / sumLikes);

        // Current posterior in probability space
        const maxLogPost = Math.max(...this.logPost);
        let post = this.logPost.map(lp => Math.exp(lp - maxLogPost));
        const sumPost = post.reduce((a, b) => a + b, 0);
        post = post.map(p => p / sumPost);

        // Exponentially weighted update
        post = post.map((p, i) => (1 - alpha) * p + alpha * likes[i]);

        // Smoothing
        post = post.map(p => (1 - this.eps) * p + this.eps * (1.0 / this.N));

        // Store back in log-space
        this.logPost = post.map(p => Math.log(p + 1e-12));

        return post;
    }

    getProb() {
        const maxLogP = Math.max(...this.logPost);
        let post = this.logPost.map(lp => Math.exp(lp - maxLogP));
        const sumPost = post.reduce((a, b) => a + b, 0);
        return post.map(p => p / sumPost);
    }
}

// -------------------------
// CRF Predictor
// -------------------------
class CRFPredictor {
    /**
     * Linear-chain CRF predictor over policy hypotheses.
     * Fixed version with proper temporal decay and adaptation.
     */
    constructor(policies, actionSpaceSize = 4, eps = 0.01, tau = 0.8, 
                pairwiseWeight = 0.3, alpha = 0.05, pSwitch = 0.02, beta = 1.0) {
        this.policies = policies;
        this.N = policies.length;
        this.actionSpaceSize = actionSpaceSize;
        this.eps = eps;
        this.tau = tau;
        this.pairwiseWeight = pairwiseWeight;
        this.alpha = alpha;
        this.pSwitch = pSwitch;
        this.beta = beta;

        this.logPost = new Array(this.N).fill(Math.log(1.0 / this.N + 1e-12));
        this.prevAction = null;
    }

    logLikelihood(state, userAction, policy) {
        /**
         * Compute log P(u | pi) with optional temporal context.
         */
        const logits = [];

        for (let a = 0; a < this.actionSpaceSize; a++) {
            // Unary potential from Q-values
            const unary = this.unaryFn(policy, state, a);

            // Pairwise potential (temporal smoothness)
            let pair = 0.0;
            if (this.prevAction !== null) {
                pair = this.pairwiseFn(this.prevAction, a);
            }

            logits.push(unary + pair);
        }

        // Softmax normalization
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(l => Math.exp(l - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(e => e / sumExp);

        return Math.log(probs[userAction] + 1e-12);
    }

    update(state, userAction) {
        /**
         * Update belief with exponential forgetting and goal persistence.
         */
        // 1. Compute log-likelihoods for all policies
        const logLikes = [];
        for (let i = 0; i < this.N; i++) {
            logLikes.push(this.logLikelihood(state, userAction, this.policies[i]));
        }

        // 2. Exponential forgetting
        this.logPost = this.logPost.map((lp, i) => (1 - this.alpha) * lp + this.alpha * logLikes[i]);

        // 3. Normalize in probability space
        const maxLog = Math.max(...this.logPost);
        let post = this.logPost.map(lp => Math.exp(lp - maxLog));
        let sumPost = post.reduce((a, b) => a + b, 0);
        post = post.map(p => p / sumPost);

        // 4. Goal-switch prior
        if (this.pSwitch > 0) {
            post = post.map(p => (1 - this.pSwitch) * p + this.pSwitch * (1.0 / this.N));
        }

        // 5. Posterior temperature
        if (this.beta !== 1.0) {
            post = post.map(p => Math.pow(p, 1.0 / this.beta));
            sumPost = post.reduce((a, b) => a + b, 0);
            post = post.map(p => p / sumPost);
        }

        // 6. Light smoothing
        post = post.map(p => (1 - this.eps) * p + this.eps * (1.0 / this.N));

        // 7. Store back in log space
        this.logPost = post.map(p => Math.log(p + 1e-12));

        // 8. Update temporal context
        this.prevAction = userAction;

        return post;
    }

    getProb() {
        const maxLog = Math.max(...this.logPost);
        let post = this.logPost.map(lp => Math.exp(lp - maxLog));
        const sumPost = post.reduce((a, b) => a + b, 0);
        return post.map(p => p / sumPost);
    }

    reset() {
        this.logPost = new Array(this.N).fill(Math.log(1.0 / this.N + 1e-12));
        this.prevAction = null;
    }

    unaryFn(policy, state, action) {
        /**
         * Unary potential from Q-values.
         */
        const Q = policy.getQValue(state, action);
        return Q / this.tau;
    }

    pairwiseFn(prevA, a) {
        /**
         * Pairwise potential encouraging temporal smoothness.
         */
        return prevA === a ? this.pairwiseWeight : 0.0;
    }
}

// -------------------------
// Shared Autonomy Policy
// -------------------------
class SharedAutoPolicy {
    constructor(policies, actionSpace) {
        this.policies = policies;
        this.actionSpace = actionSpace;
        
        // Pre-compute normalization parameters for each policy
        this.qMins = [];
        this.qMaxs = [];
        
        for (const policy of policies) {
            let min = Infinity;
            let max = -Infinity;
            
            for (let i = 0; i < policy.qTable.length; i++) {
                for (let j = 0; j < policy.qTable[i].length; j++) {
                    for (let k = 0; k < policy.qTable[i][j].length; k++) {
                        const val = policy.qTable[i][j][k];
                        if (val < min) min = val;
                        if (val > max) max = val;
                    }
                }
            }
            
            this.qMins.push(min);
            this.qMaxs.push(max);
        }
    }

    normalizeQValue(qValue, policyIdx) {
        /**
         * Normalize Q-value to [0, 1] based on the policy's Q-table range
         */
        const qMin = this.qMins[policyIdx];
        const qMax = this.qMaxs[policyIdx];
        
        if (qMax - qMin === 0) {
            return 0.5;
        }
        return (qValue - qMin) / (qMax - qMin);
    }

    getAction(state, probPolicy, returnDist = false, sample = false) {
        const actions = this.actionSpace;
        const qs = [];
        
        // Get Q-values and normalize them
        for (let i = 0; i < this.policies.length; i++) {
            const rawQs = [];
            for (const a of actions) {
                rawQs.push(this.policies[i].getQValue(state, a));
            }
            const normalizedQs = rawQs.map(q => this.normalizeQValue(q, i));
            qs.push(normalizedQs);
        }
        
        // Compute expected Q-values
        const expectedQ = new Array(actions.length).fill(0);
        for (let a = 0; a < actions.length; a++) {
            for (let i = 0; i < this.policies.length; i++) {
                expectedQ[a] += probPolicy[i] * qs[i][a];
            }
        }
        
        // Apply logsumexp normalization
        const maxQ = Math.max(...expectedQ);
        const normalizedQ = expectedQ.map(q => q - maxQ);
        const expQ = normalizedQ.map(q => Math.exp(q));
        const sumExpQ = expQ.reduce((a, b) => a + b, 0);
        
        if (returnDist) {
            const dist = expQ.map(q => q / sumExpQ);
            const maxIdx = expectedQ.indexOf(Math.max(...expectedQ));
            return [actions[maxIdx], dist];
        } else if (sample) {
            const dist = expQ.map(q => q / sumExpQ);
            const rand = Math.random();
            let cumSum = 0;
            for (let i = 0; i < dist.length; i++) {
                cumSum += dist[i];
                if (rand <= cumSum) {
                    return actions[i];
                }
            }
            return actions[actions.length - 1];
        } else {
            const maxIdx = expectedQ.indexOf(Math.max(...expectedQ));
            return actions[maxIdx];
        }
    }
}

// -------------------------
// Main Simulator Class
// -------------------------
class DotSimulator {
    /**
     * Canvas-based dot mover with shared-autonomy inference, assistance, arbitration.
     * 
     * This refactor keeps all behavior identical to the provided implementation:
     * - Same defaults for sizes, speeds, enums
     * - Same event flow, drawing, and blending behaviors
     * - Same file loading and policy-color mapping
     */
    constructor(policyDir = "trained_policies", 
                inferenceType = Inference.BAYESIAN,
                assistanceType = Assistance.DISTRIBUTION,
                arbitrationType = Arbitration.LINEAR) {
        
        // Configuration
        this.GAMMA = DEFAULTS.GAMMA;
        this.SCREEN_WIDTH = DEFAULTS.SCREEN_WIDTH;
        this.SCREEN_HEIGHT = DEFAULTS.SCREEN_HEIGHT;
        this.DOT_RADIUS = DEFAULTS.DOT_RADIUS;
        this.DOT_SPEED = DEFAULTS.DOT_SPEED;
        this.GRID_SIZE = DEFAULTS.GRID_SIZE;
        this.TEXT_SIZE = DEFAULTS.TEXT_SIZE;

        // Enums
        this.INFERENCE_TYPE = inferenceType;
        this.ASSISTANCE_TYPE = assistanceType;
        this.ARBITRATION_TYPE = arbitrationType;

        // Colors
        this.BLACK = '#000000';
        this.WHITE = '#FFFFFF';

        // State
        this.dotX = this.SCREEN_WIDTH / 2;
        this.dotY = this.SCREEN_HEIGHT / 2;
        this.ACTION_SPACE_LEN = 4;
        this.lastClickTime = 0;

        // Canvas setup
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');

        // Policies (will be loaded)
        this.POLICY_DIR = policyDir;
        this.POLICY_FILES = [];
        this.POLICIES = [];
        this.prob = [];
        this.POLICY_COLORS = [];

        // Background images
        this.AGENT_IMG_TRUE = false;
        this.backgroundImages = {};
        this.agentImage = null;

        // Keyboard state
        this.keys = {};
        
        // Robot confidence (for probabilistic arbitration)
        this.robotConfidence = 0.0;

        this.setupEventListeners();
    }

    setConditions(condition) {
    /**
     * Based on the current study condition, set our variables accordingly
     */
    if (condition=='A') {
        console.log("Setting conditions")
        this.inferenceType = Inference.BAYESIAN,
        this.assistanceType = Assistance.DISTRIBUTION,
        this.arbitrationType = Arbitration.LINEAR
    }
    console.log(this.inferenceType)
    console.log(this.assistanceType)
    console.log(this.arbitrationType)
}
    // -------------------------
    // Background image loading
    // -------------------------
    loadBackgroundImages(backgroundData) {
        /**
         * Load background images and positions from data structure.
         * Expected format: [{filename, x, y, xscale, yscale}, ...]
         */
        for (const bgConfig of backgroundData) {
            const filename = bgConfig.filename;
            const x = bgConfig.x;
            const y = bgConfig.y;
            const xscale = bgConfig.xscale;
            const yscale = bgConfig.yscale;
            
            // Check if this is the agent image (cursor)
            if (x === 'None' || x === null) {
                console.log("Loading cursor image:", filename);
                this.AGENT_IMG_TRUE = true;
                const img = new Image();
                img.onload = () => {
                    this.agentImage = {
                        img: img,
                        width: xscale,
                        height: yscale,
                        loaded: true
                    };
                    console.log("Cursor image loaded successfully");
                };
                img.onerror = () => {
                    console.warn(`Warning: agent image not found: ${filename}`);
                    this.AGENT_IMG_TRUE = false;
                };
                img.src = `background_images/${filename}`;
                continue;
            }
            
            // Load regular background image
            const img = new Image();
            img.onload = () => {
                this.backgroundImages[filename] = {
                    image: img,
                    pos: [parseFloat(x), parseFloat(y)],
                    width: xscale,
                    height: yscale,
                    loaded: true
                };
                console.log(`Background image loaded: ${filename}`);
            };
            img.onerror = () => {
                console.warn(`Warning: background image not found: ${filename}`);
            };
            img.src = `background_images/${filename}`;
        }
    }

    drawAgent(x, y) {
        if (this.agentImage && this.agentImage.loaded) {
            const imgRect = {
                x: x - this.agentImage.width / 2,
                y: y - this.agentImage.height / 2
            };
            this.ctx.drawImage(this.agentImage.img, imgRect.x, imgRect.y, 
                             this.agentImage.width, this.agentImage.height);
        }
    }

    drawBackgrounds() {
        /**
         * Draw all background images at their specified positions.
         */
        for (const filename in this.backgroundImages) {
            const data = this.backgroundImages[filename];
            if (data.loaded) {
                const imgRect = {
                    x: data.pos[0] - data.width / 2,
                    y: data.pos[1] - data.height / 2
                };
                this.ctx.save();
                this.ctx.globalAlpha = 0.3;
                this.ctx.drawImage(data.image, imgRect.x, imgRect.y, data.width, data.height);
                this.ctx.restore();
            }
        }
    }

    // -------------------------
    // Utility & transformation
    // -------------------------
    getState(x, y) {
        /**
         * Map continuous (x,y) to a discrete grid state (sx, sy).
         */
        const sx = Math.floor(Math.max(0, Math.min(x, this.SCREEN_WIDTH - 1)) / this.GRID_SIZE);
        const sy = Math.floor(Math.max(0, Math.min(y, this.SCREEN_HEIGHT - 1)) / this.GRID_SIZE);
        return [sx, sy];
    }

    indexToTuple(index) {
        /**
         * Return direction vector for action index. Unknown -> [0,0].
         */
        return INDEX_TO_TUPLE[index] || [0, 0];
    }

    executeAction(action) {
        /**
         * Apply a unit-direction action (dx,dy) scaled by DOT_SPEED to the dot.
         */
        this.dotX += this.DOT_SPEED * action[0];
        this.dotY += this.DOT_SPEED * action[1];
    }

    ensureWithinBoundaries() {
        /**
         * Clamp the dot to the visible gameplay region.
         */
        this.dotX = Math.max(this.DOT_RADIUS, Math.min(this.SCREEN_WIDTH - this.DOT_RADIUS, this.dotX));
        this.dotY = Math.max(this.DOT_RADIUS, Math.min(this.SCREEN_HEIGHT - this.DOT_RADIUS, this.dotY));
    }

    redrawScreen() {
        /**
         * Clear and redraw everything (backgrounds + dot).
         */
        // Fill background
        this.ctx.fillStyle = this.BLACK;
        this.ctx.fillRect(0, 0, this.SCREEN_WIDTH, this.SCREEN_HEIGHT);

        // Draw backgrounds FIRST
        this.drawBackgrounds();

        // Draw the dot
        const dotScreenY = this.dotY;
        if (this.AGENT_IMG_TRUE) {
            this.drawAgent(this.dotX, dotScreenY);
        } else {
            this.ctx.fillStyle = this.WHITE;
            this.ctx.beginPath();
            this.ctx.arc(this.dotX, dotScreenY, this.DOT_RADIUS, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    // -------------------------
    // Core loop and helpers
    // -------------------------
    createPredictor() {
        const PREDICTOR_MAP = {
            [Inference.BAYESIAN]: BayesianPredictor,
            [Inference.MAX_ENT]: MaxEntPredictor,
            [Inference.CRF]: CRFPredictor,
        };
        const cls = PREDICTOR_MAP[this.INFERENCE_TYPE] || BayesianPredictor;
        return new cls(this.POLICIES);
    }

    createAssistant() {
        return new SharedAutoPolicy(this.POLICIES, Array.from({length: this.ACTION_SPACE_LEN}, (_, i) => i));
    }

    blend(u, a) {
        /**
         * Combine user command u (index) and robot action a (index) according to arbitration:
         * - LINEAR: linear blend by GAMMA (then normalize)
         * - PROBABILISTIC: weight by robot confidence (max of this.prob)
         * - ONLY_USER: return user vector
         */
        const uVec = this.indexToTuple(u);
        const aVec = this.indexToTuple(a);

        if (this.ARBITRATION_TYPE === Arbitration.ONLY_USER) {
            return uVec;
        }

        let blended;
        if (this.ARBITRATION_TYPE === Arbitration.LINEAR) {
            blended = [
                uVec[0] * this.GAMMA + aVec[0] * (1 - this.GAMMA),
                uVec[1] * this.GAMMA + aVec[1] * (1 - this.GAMMA)
            ];
        } else if (this.ARBITRATION_TYPE === Arbitration.PROBABILISTIC) {
            this.robotConfidence = this.prob.length ? Math.max(...this.prob) : 0.0;
            const pRobot = this.robotConfidence;
            blended = [
                pRobot * aVec[0] + (1 - pRobot) * uVec[0],
                pRobot * aVec[1] + (1 - pRobot) * uVec[1]
            ];
        } else {
            blended = uVec; // fallback
        }

        const mag = Math.sqrt(blended[0] * blended[0] + blended[1] * blended[1]);
        if (mag === 0) {
            return [0, 0];
        }
        return [blended[0] / mag, blended[1] / mag];
    }

    setupEventListeners() {
        document.addEventListener('keydown', (e) => {
            this.keys[e.key] = true;
        });

        document.addEventListener('keyup', (e) => {
            this.keys[e.key] = false;
        });

        // document.getElementById('inferenceType').addEventListener('change', (e) => {
        //     this.INFERENCE_TYPE = e.target.value;
        //     this.reset();
        // });

        // document.getElementById('arbitrationType').addEventListener('change', (e) => {
        //     this.ARBITRATION_TYPE = e.target.value;
        // });

        // document.getElementById('gammaSlider').addEventListener('input', (e) => {
        //     this.GAMMA = parseFloat(e.target.value);
        //     document.getElementById('gammaValue').textContent = this.GAMMA.toFixed(1);
        // });

        // document.getElementById('resetBtn').addEventListener('click', () => {
        //     this.reset();
        // });
    }

    reset() {
        this.dotX = this.SCREEN_WIDTH / 2;
        this.dotY = this.SCREEN_HEIGHT / 2;
        this.prob = new Array(this.POLICIES.length).fill(1.0 / this.POLICIES.length);
        this.robotConfidence = 0.0;
        this.predictor = this.createPredictor();
        this.updateUI();
    }

    updateUI() {
        if (this.prob.length > 0) {
            const probText = this.prob.map((p, i) => `P${i + 1}: ${(p * 100).toFixed(1)}%`).join(', ');
            // document.getElementById('probabilities').textContent = `Policy Probabilities: ${probText}`;
            // document.getElementById('robotConfidence').textContent = `Robot Confidence: ${(this.robotConfidence * 100).toFixed(1)}%`;
        }
    }

    initializePolicies(policies) {
        this.POLICIES = policies;
        this.POLICY_FILES = policies.map(p => p.qTableFile);
        this.prob = new Array(policies.length).fill(1.0 / policies.length);
        this.POLICY_COLORS = generateColors(policies.length);
        
        this.predictor = this.createPredictor();
        this.assistant = this.createAssistant();
        
        this.updateUI();
    }

    runShared() {
        /**
         * Main loop: handle input, inference, assistance, arbitration, drawing.
         */
        const gameLoop = () => {
            // Check user action for THIS frame
            let u = -1;
            if (this.keys['ArrowUp']) {
                u = 0;
            } else if (this.keys['ArrowDown']) {
                u = 1;
            } else if (this.keys['ArrowLeft']) {
                u = 2;
            } else if (this.keys['ArrowRight']) {
                u = 3;
            }

            // If no user action: robot does nothing
            if (u === -1) {
                this.redrawScreen();
                requestAnimationFrame(gameLoop);
                return;
            }

            // Inference
            this.prob = this.predictor.update(this.getState(this.dotX, this.dotY), u);

            // Assistance
            const optimalAction = this.assistant.getAction(
                this.getState(this.dotX, this.dotY),
                this.prob
            );

            // Arbitration & Execution
            const blendedAction = this.blend(u, optimalAction);
            this.executeAction(blendedAction);

            // Boundaries & draw
            this.ensureWithinBoundaries();
            this.redrawScreen();
            
            this.updateUI();

            requestAnimationFrame(gameLoop);
        };

        gameLoop();
    }
}

// -------------------------
// Synthetic Q-table generation for demo
// -------------------------
function createTopLeftPolicy(gridSize, actionSize) {
    const qTable = [];
    for (let x = 0; x < gridSize; x++) {
        qTable[x] = [];
        for (let y = 0; y < gridSize; y++) {
            qTable[x][y] = [];
            const dist = Math.sqrt(x * x + y * y);
            
            qTable[x][y][0] = y > 0 ? 100 - dist : -50; // UP
            qTable[x][y][1] = y < gridSize - 1 ? -50 : -100; // DOWN
            qTable[x][y][2] = x > 0 ? 100 - dist : -50; // LEFT
            qTable[x][y][3] = x < gridSize - 1 ? -50 : -100; // RIGHT
        }
    }
    return qTable;
}

function createBottomRightPolicy(gridSize, actionSize) {
    const qTable = [];
    const target = gridSize - 1;
    
    for (let x = 0; x < gridSize; x++) {
        qTable[x] = [];
        for (let y = 0; y < gridSize; y++) {
            qTable[x][y] = [];
            const dist = Math.sqrt((x - target) ** 2 + (y - target) ** 2);
            
            qTable[x][y][0] = y < target ? -50 : -100; // UP
            qTable[x][y][1] = y < target ? 100 - dist : -50; // DOWN
            qTable[x][y][2] = x < target ? -50 : -100; // LEFT
            qTable[x][y][3] = x < target ? 100 - dist : -50; // RIGHT
        }
    }
    return qTable;
}

function createOrbitPolicy(gridSize, actionSize) {
    const qTable = [];
    const center = gridSize / 2;
    const targetRadius = gridSize / 3;
    
    for (let x = 0; x < gridSize; x++) {
        qTable[x] = [];
        for (let y = 0; y < gridSize; y++) {
            qTable[x][y] = [];
            
            const dx = x - center;
            const dy = y - center;
            const currentRadius = Math.sqrt(dx * dx + dy * dy);
            const radiusError = Math.abs(currentRadius - targetRadius);
            
            const tangentX = -dy;
            const tangentY = dx;
            
            qTable[x][y][0] = (tangentY < 0 ? 50 : -20) - radiusError * 2; // UP
            qTable[x][y][1] = (tangentY > 0 ? 50 : -20) - radiusError * 2; // DOWN
            qTable[x][y][2] = (tangentX < 0 ? 50 : -20) - radiusError * 2; // LEFT
            qTable[x][y][3] = (tangentX > 0 ? 50 : -20) - radiusError * 2; // RIGHT
        }
    }
    return qTable;
}

// -------------------------
// Initialization
// -------------------------
async function loadPolicies() {
    const gridSize = 30; // 600 / 20
    const actionSize = 4;
    
    const policies = [];
    const policyNames = ['topleft', 'bottomright', 'orbit'];
    
    for (const name of policyNames) {
        try {
            const response = await fetch(`q_table_${name}.json`);
            if (response.ok) {
                const qTable = await response.json();
                policies.push(new DotPolicy(`q_table_${name}.json`, qTable));
            } else {
                throw new Error('Failed to load');
            }
        } catch (e) {
            console.warn(`Failed to load policy ${name}, creating synthetic policy`);
            let qTable;
            if (name === 'topleft') {
                qTable = createTopLeftPolicy(gridSize, actionSize);
            } else if (name === 'bottomright') {
                qTable = createBottomRightPolicy(gridSize, actionSize);
            } else {
                qTable = createOrbitPolicy(gridSize, actionSize);
            }
            policies.push(new DotPolicy(`q_table_${name}.json`, qTable));
        }
    }
    
    return policies;
}

// -------------------------
// Main entry point
// -------------------------
async function main() {
    console.log("Entering main")
    const policies = await loadPolicies();
    const simulator = new DotSimulator(
        "trained_policies"
    );

    
    // Get the study condition
    const condition = document.getElementById('condition').innerText;
    simulator.setConditions(condition)
    
    // Load background images
    const backgroundConfig = [
        {filename: 'sugar.png', x: 55, y: 545, xscale: 150, yscale: 150},
        {filename: 'milk_carton.png', x: 545, y: 545, xscale: 150, yscale: 150},
        {filename: 'cheese.png', x: 545, y: 55, xscale: 150, yscale: 150},
        {filename: 'egg.png', x: 55, y: 55, xscale: 150, yscale: 150},
        {filename: 'mixing_bowl.png', x: 300, y: 300, xscale: 300, yscale: 300},
        {filename: 'spoon.png', x: 'None', y: 'None', xscale: 80, yscale: 80}
    ];
    // "bayesian", "maxent", "crf"
    // simulator.INFERENCE_TYPE = "maxent";
    // "linear", "probabilistic", "onlyuser"
    // simulator.ARBITRATION_TYPE = "linear";
    
    simulator.loadBackgroundImages(backgroundConfig);
    simulator.initializePolicies(policies);
    simulator.runShared();
}

// Start when page loads
window.addEventListener('load', main);
