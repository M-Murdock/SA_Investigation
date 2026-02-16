# #!/usr/bin/env python3
# import numpy as np
# import scipy

# from util import iter_space


# class SharedAutoPolicy:
#     def __init__(self, policies, action_space):
#         self._action_space = action_space
#         self._policies = policies
#         # TODO (somehow): check that action spaces are compatible and discrete!
    
#     def get_action(self, x, prob_policy, return_dist=False, sample=False):
#         actions = tuple(iter_space(self._action_space))
#         qs = np.zeros((len(self._policies), len(actions)))
#         for i, policy in enumerate(self._policies):
#             if hasattr(policy, "get_q_values"):
#                 qs[i] = policy.get_q_values(x, actions)
#             else:
#                 qs[i] = [ policy.get_q_value(x, a) for a in actions]
#         expected_q = np.dot(prob_policy, qs)
#         expected_q -= scipy.special.logsumexp(expected_q)

#         if return_dist:
#             return actions[np.argmax(expected_q)], np.exp(expected_q)
#         elif sample:
#             return np.random.choice(actions, p=np.exp(expected_q))
#         else:
#             return actions[np.argmax(expected_q)]


# if __name__ == "__main__":
#     SharedAutoPolicy(None, None)
#!/usr/bin/env python3
import numpy as np
import scipy

from util import iter_space


class SharedAutoPolicy:
    def __init__(self, policies, action_space):
        self._action_space = action_space
        self._policies = policies
        # TODO (somehow): check that action spaces are compatible and discrete!
        
        # PRE-COMPUTE normalization parameters for each policy
        self.q_mins = []
        self.q_maxs = []
        for policy in policies:
            q_min = np.min(policy.q_table)
            q_max = np.max(policy.q_table)
            self.q_mins.append(q_min)
            self.q_maxs.append(q_max)
    
    def normalize_q_value(self, q_value, policy_idx):
        """Normalize Q-value to [0, 1] based on the policy's Q-table range"""
        q_min = self.q_mins[policy_idx]
        q_max = self.q_maxs[policy_idx]
        
        if q_max - q_min == 0:
            return 0.5
        return (q_value - q_min) / (q_max - q_min)
    
    def get_action(self, x, prob_policy, return_dist=False, sample=False):
        actions = tuple(iter_space(self._action_space))
        qs = np.zeros((len(self._policies), len(actions)))
        
        # Get Q-values and normalize them
        for i, policy in enumerate(self._policies):
            if hasattr(policy, "get_q_values"):
                raw_qs = policy.get_q_values(x, actions)
            else:
                raw_qs = [policy.get_q_value(x, a) for a in actions]
            
            # Normalize Q-values for this policy
            qs[i] = [self.normalize_q_value(q, i) for q in raw_qs]
        
        expected_q = np.dot(prob_policy, qs)
        expected_q -= scipy.special.logsumexp(expected_q)

        if return_dist:
            return actions[np.argmax(expected_q)], np.exp(expected_q)
        elif sample:
            return np.random.choice(actions, p=np.exp(expected_q))
        else:
            return actions[np.argmax(expected_q)]


if __name__ == "__main__":
    SharedAutoPolicy(None, None)