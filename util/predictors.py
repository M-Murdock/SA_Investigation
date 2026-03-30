import numpy as np

class BayesianPredictor:
    # def __init__(self, policies, action_space_size=4, prior=None, tau=0.8, eps=1e-3):
    def __init__(self, policies, action_space_size=4, prior=None, tau=0.8, eps=1e-3):
        self.policies = policies
        self.N = len(policies)
        self.action_space_size = action_space_size
        
        # smoothing hyperparameter for posterior
        self.eps = eps  
        self.tau = tau  # likelihood temperature

        # log posterior
        if prior is None:
            prior = np.ones(self.N) / self.N
        
        self.log_post = np.log(prior + 1e-12)

    def log_likelihood(self, state, user_action, policy):
        """Return log P(u | pi)."""
        Q = np.array([policy.get_q_value(state, a) for a in range(self.action_space_size)])

        # softmax likelihood P(u | pi)
        logits = Q / self.tau
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # return np.log(probs[user_action] + 1e-12)
        return np.log(probs[user_action] + 1e-8)

    def update( 
            self,
            state,
            user_action,
            alpha=0.05,        # forgetting / adaptation rate
            p_switch=0.02,     # goal-switch prior
            beta=1           # posterior temperature (>1 = smoother)
        ):
        """
        Bayesian update with forgetting, goal persistence, and posterior smoothing.
        """

        # 1. Compute log-likelihoods
        log_likes = np.zeros(self.N)
        for i, pi in enumerate(self.policies):
            log_likes[i] = self.log_likelihood(state, user_action, pi)

        # 2. Exponential forgetting (key fix)
        # Blends past belief with current evidence
        self.log_post = (1 - alpha) * self.log_post + alpha * log_likes

        # 3. Normalize in probability space
        max_logp = np.max(self.log_post)
        post = np.exp(self.log_post - max_logp)
        post /= np.sum(post)

        # 4. Goal-switch prior (intent persistence)
        if p_switch > 0:
            post = (1 - p_switch) * post + p_switch * (1.0 / self.N)

        # 5. Posterior temperature (optional but useful)
        if beta != 1.0:
            post = post ** (1.0 / beta)
            post /= np.sum(post)

        # 6. Light smoothing (numerical safety only)
        post = (1 - self.eps) * post + self.eps * (1.0 / self.N)

        # 7. Store back in log space
        self.log_post = np.log(post + 1e-12)

        return post


    def get_prob(self):
        max_logp = np.max(self.log_post)
        post = np.exp(self.log_post - max_logp)
        return post / np.sum(post)


class MaxEntPredictor:
    def __init__(self, policies, action_space_size=4, tau=0.8, eps=1e-2):
        """
        policies: list of policies to evaluate
        action_space_size: number of discrete actions
        tau: softmax temperature
        eps: smoothing to prevent zeros
        """
        self.policies = policies
        self.N = len(policies)
        self.action_space_size = action_space_size
        self.tau = tau
        self.eps = eps

        # initialize uniform belief over policies
        self.log_post = np.log(np.ones(self.N) / self.N + 1e-12)

    def log_likelihood(self, state, user_action, policy):
        """MaxEnt IOC likelihood: P(u | pi) proportional to exp(Q(s,u)/tau)."""
        Q = np.array([policy.get_q_value(state, a) for a in range(self.action_space_size)])
        logits = Q / self.tau
        logits -= np.max(logits)  # for numerical stability
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.log(probs[user_action] + 1e-12)

    def update(self, state, user_action, alpha=0.5):
        """
        Update the belief over policies using MaxEnt likelihood.
        alpha: learning rate for smoothing the posterior
        """
        log_likes = np.zeros(self.N)
        for i, pi in enumerate(self.policies):
            log_likes[i] = self.log_likelihood(state, user_action, pi)

        # Convert to probability
        likes = np.exp(log_likes - np.max(log_likes))
        likes /= np.sum(likes)

        # Current posterior in probability space
        post = np.exp(self.log_post - np.max(self.log_post))
        post /= np.sum(post)

        # Exponentially weighted update
        post = (1 - alpha) * post + alpha * likes

        # smoothing
        post = (1 - self.eps) * post + self.eps * (1.0 / self.N)

        # store back in log-space
        self.log_post = np.log(post + 1e-12)

        return post


    def get_prob(self):
        max_logp = np.max(self.log_post)
        post = np.exp(self.log_post - max_logp)
        return post / np.sum(post)


class CRFPredictor:
    """
    CRF-inspired online predictor over policy hypotheses for shared autonomy.

    Uses unary potentials (Q-values) and policy-aware pairwise potentials
    to define an emission model, then performs online Bayesian filtering
    with exponential forgetting to maintain a belief over user goals.
    """

    def __init__(
        self,
        policies,
        action_space_size=4,
        eps=0.01,
        tau=0.8,
        pairwise_weight=0.3,
        alpha=0.05,
        p_switch=0.02,
        beta=1.0,
    ):
        self.policies = policies
        self.N = len(policies)
        self.action_space_size = action_space_size

        self.eps = eps
        self.tau = tau
        self.pairwise_weight = pairwise_weight
        self.alpha = alpha          # forgetting rate: decays old evidence
        self.p_switch = p_switch    # goal-switch prior (uniform mixing)
        self.beta = beta            # posterior temperature

        self.log_post = np.log(np.ones(self.N) / self.N + 1e-12)
        self.prev_action = None

    def log_likelihood(self, state, user_action, policy):
        """
        Compute log P(user_action | state, policy) using CRF-style potentials.

        The pairwise potential is now policy-aware: it rewards action repetition
        only when the policy itself would prefer that action, so it actually
        helps discriminate between goals rather than boosting all policies equally.
        """
        logits = np.zeros(self.action_space_size)

        for a in range(self.action_space_size):
            # Unary potential: how much this policy values action a in this state
            unary = self.unary_fn(policy, state, a)

            # Pairwise potential: policy-aware temporal consistency
            pair = 0.0
            if self.prev_action is not None:
                pair = self.pairwise_fn(policy, state, self.prev_action, a)

            logits[a] = unary + pair

        # Softmax normalization
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)

        return np.log(probs[user_action] + 1e-12)

    def update(self, state, user_action):
        """
        Bayesian belief update with exponential forgetting and goal persistence.

        Key fix: old evidence is decayed by (1 - alpha), but new evidence
        (log-likelihoods) is added at full strength rather than scaled by alpha.
        This gives proper online filtering where recent observations have full
        impact while old evidence gradually fades.
        """
        # 1. Compute log-likelihoods for all policies
        log_likes = np.array([
            self.log_likelihood(state, user_action, pi)
            for pi in self.policies
        ])

        # 2. Bayesian update with exponential forgetting:
        #    - Decay accumulated log-posterior (old evidence fades)
        #    - Add new log-likelihood at full strength
        self.log_post = (1 - self.alpha) * self.log_post + log_likes

        # 3. Normalize in probability space
        max_log = np.max(self.log_post)
        post = np.exp(self.log_post - max_log)
        post /= np.sum(post)

        # 4. Goal-switch prior: mix with uniform to allow intent changes.
        #    This subsumes the old eps-smoothing, so we only need one
        #    uniform-mixing step. Use whichever rate is larger.
        mix_rate = max(self.p_switch, self.eps)
        post = (1 - mix_rate) * post + mix_rate * (1.0 / self.N)

        # 5. Posterior temperature (optional sharpening/flattening)
        if self.beta != 1.0:
            post = post ** (1.0 / self.beta)
            post /= np.sum(post)

        # 6. Store back in log space
        self.log_post = np.log(post + 1e-12)

        # 7. Update temporal context
        self.prev_action = user_action

        return post

    def get_prob(self):
        """Return current posterior distribution."""
        max_log = np.max(self.log_post)
        post = np.exp(self.log_post - max_log)
        return post / np.sum(post)

    def reset(self):
        """Reset belief and temporal context."""
        self.log_post = np.log(np.ones(self.N) / self.N + 1e-12)
        self.prev_action = None

    def unary_fn(self, policy, state, action):
        """Unary potential: Q-value scaled by temperature."""
        Q = policy.get_q_value(state, action)
        return Q / self.tau

    def pairwise_fn(self, policy, state, prev_a, a):
        """
        Policy-aware pairwise potential for temporal consistency.

        Instead of a flat bonus for repeating any action (which helps no
        policy more than another), this rewards repetition proportional to
        how much the *policy* values the repeated action. This makes the
        pairwise term discriminative: if the user repeats an action that
        policy_i strongly prefers, policy_i gets a bigger emission
        probability boost than policy_j which doesn't value that action.
        """
        if prev_a == a:
            # Scale the persistence bonus by how much this policy wants
            # the repeated action (normalized Q-value as a soft indicator)
            q_val = policy.get_q_value(state, a) / self.tau
            return self.pairwise_weight * max(q_val, 0.0)
        return 0.0