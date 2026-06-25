from .builtins import *
from .external import External

from typing import Iterable
from deepsoftlog.algebraic_prover.terms.expression import Clause, Expr, Fact
from deepsoftlog.algebraic_prover.terms.probability_annotation import ProbabilisticFact, LogProbabilisticExpr, ProbabilisticExpr
from deepsoftlog.logic.soft_unify import get_unify_fact, is_soft

import torch

ALL_BUILTINS = (
    External("is", 2, builtin_is),
    External("==", 2, builtin_eq),
    External("\\==", 2, builtin_neq),
    External("writeln", 1, builtin_writeln),
    External("fresh", 1, builtin_fresh),
)

        
        
#         # Extract inner terms if they're soft
        
#             # Get soft unification probability
            
#             # Create a fact with EMPTY BODY - this is key
            
#             # Create a probabilistic FACT (not a rule)
#                 prob,  # The soft unification probability
#                 ":-",  # This makes it a clause
#                 type_expr,  # The head
#                 empty_body,  # Empty body = fact, not rule
#             )
            
#             # Simple version: just yield a Fact with empty substitution
#             yield Fact(type_expr), {}, set()
            
#             # Still succeed
#             yield Fact(Expr("type", t1, t2)), {}, set()

        
#         """Type predicate that directly adds soft unification facts"""
            
#             # Extract inner terms
            
#                 # Get soft unification fact
                
#                 # CRITICAL: Return the soft_fact in the third position of the tuple
#                 # This is what adds it to the proof's soft facts
#                 yield Fact(Expr("type", t1, t2)), {}, {soft_fact}
#                 # Still succeed but without soft facts
#                 yield Fact(Expr("type", t1, t2)), {}, set()
#             # For non-soft terms, just succeed
#             yield Fact(Expr("type", t1, t2)), {}, set()

        
#         """Type predicate that directly creates a probabilistic fact"""
            
#             # Extract inner terms
            
#                 # Calculate similarity
                
#                 # Create type fact
                
#                 # Create probabilistic fact with exact similarity as probability
                
#                 # Return this directly - no soft facts needed
#                 yield prob_fact, {}, set()
                
#                 (f"Error in TypeExternal: {e}")
#                 # Create a low-probability fact on error
#                 yield ProbabilisticFact(0.01, Expr("type", t1, t2)), {}, set()
#             # For non-soft terms, just succeed with high probability
#             yield ProbabilisticFact(0.99, Expr("type", t1, t2)), {}, set()


        
            
#             # Extract inner terms
            
#                 # Get log similarity
                
#                 # Create type fact
                
#                 # Create LogProbabilisticExpr directly with the log probability
#                     log_similarity,  # Use log probability directly
#                     ":-",           # Create a fact
#                     type_fact,      # The type predicate
#                 )
                
#                 yield log_prob_fact, {}, set()
                
#                 # Use a small but valid log probability (-4.6 ≈ log(0.01))
#                 yield Fact(LogProbabilisticExpr(-4.6, ":-", Expr("type", t1, t2), Expr(","), infix=True), {}, set())
#             # For non-soft terms, use log(0.99) ≈ -0.01
#             yield Fact(LogProbabilisticExpr(-0.01, ":-", Expr("type", t1, t2), Expr(","), infix=True), {}, set())

class TypeExternal(External):
    def __init__(self, store_getter, metric="l2"):
        super().__init__("type", 2, None)
        self.store_getter = store_getter
        self.metric = metric
        self.cache = {}  # Cache for previously computed scores
        
    #     """
    #     Implements type/2 predicate that performs soft unification between terms.
    #     """
        
    #     # Create a cache key for these terms
        
    #     # Check if we've seen this pair before
        
    #         # Extract inner terms if they're soft terms
            
    #         # Default to a high log probability for identical terms
            
    #         # Calculate soft unification score if terms are different
    #                 # Fallback to a low log probability
            
    #         # Create the fact expression
            
    #         # Create a log probabilistic fact using the imported constructor
    #         # We need to create a fact with the structure:
    #         # where body is an empty conjunction (representing true)
            
    #             log_score,  # The log probability
    #             ":-",       # Rule functor
    #             type_expr,  # Head of the rule
    #         )
            
    #         # Cache for future use
            
    #         # Return the fact expression, empty substitution, and the set with our soft fact
            
    #         # Return a simple success with very low probability on error
    
    # Debug for TypeExternal.get_answers method
    def get_answers(self, t1, t2) -> Iterable[tuple[Clause, dict, set]]:
        """
        Debug instrumented version to track gradient flow
        """
        
        store = self.store_getter()
        
        # Extract inner terms if they're soft terms
        t1_inner = t1.arguments[0] if hasattr(t1, 'arguments') and is_soft(t1) else t1
        t2_inner = t2.arguments[0] if hasattr(t2, 'arguments') and is_soft(t2) else t2
        
        # Default to a high log probability for identical terms
        log_score = 0.0  # log(1.0) = 0.0
        
        # Calculate soft unification score if terms are different
        if t1_inner != t2_inner:
            try:
                # Track embedding retrieval
                
                # Get the log similarity score
                log_score = store.soft_unify_score(t1_inner, t2_inner, self.metric)
                if isinstance(log_score, torch.Tensor):
                    pass
                    
            except Exception as e:
                log_score = -10.0  # Fallback
        
        # Create the fact expression
        type_expr = Expr("type", t1, t2)
        
        # Create a log probabilistic fact
        soft_fact = LogProbabilisticExpr(
            log_score,  # The log probability
            ":-",       # Rule functor
            type_expr,  # Head of the rule
            Expr(","),  # Empty body (true)
            infix=True  # Use infix notation
        )
        
        if hasattr(soft_fact, 'get_log_probability'):
            log_prob = soft_fact.get_log_probability()
            if isinstance(log_prob, torch.Tensor):
                pass
        
        # Return the fact expression, empty substitution, and soft fact
        return [(Fact(type_expr), {}, {soft_fact})]