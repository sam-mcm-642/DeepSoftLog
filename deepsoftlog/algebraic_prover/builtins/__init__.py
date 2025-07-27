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

# class TypeBuiltin(External):
#     def __init__(self, get_store_fn, metric):
#         super().__init__("type", 2, None)
#         self.get_store = get_store_fn
#         self.metric = metric
        
#     def get_answers(self, t1, t2) -> Iterable[tuple[Clause, dict, set]]:
#         store = self.get_store()
        
#         # Extract inner terms if they're soft
#         t1_inner = t1.arguments[0] if is_soft(t1) else t1
#         t2_inner = t2.arguments[0] if is_soft(t2) else t2
        
#         try:
#             # Get soft unification probability
#             prob = 0.99 if t1_inner == t2_inner else store.soft_unify_score(t1_inner, t2_inner, self.metric)
            
#             # Create a fact with EMPTY BODY - this is key
#             type_expr = Expr("type", t1, t2)
#             empty_body = Expr(",", infix=True)  # Empty conjunction = true
            
#             # Create a probabilistic FACT (not a rule)
#             prob_fact = LogProbabilisticExpr(
#                 prob,  # The soft unification probability
#                 ":-",  # This makes it a clause
#                 type_expr,  # The head
#                 empty_body,  # Empty body = fact, not rule
#                 infix=True
#             )
            
#             # Simple version: just yield a Fact with empty substitution
#             yield Fact(type_expr), {}, set()
            
#         except Exception as e:
#             print(f"Error in TypeBuiltin: {e}")
#             # Still succeed
#             yield Fact(Expr("type", t1, t2)), {}, set()

# class TypeExternal(External):
#     def __init__(self, store_getter, metric):
#         super().__init__("type", 2, None)
#         self.store_getter = store_getter
#         self.metric = metric
        
#     def get_answers(self, t1, t2) -> Iterable[tuple[Clause, dict, set]]:
#         """Type predicate that directly adds soft unification facts"""
#         if is_soft(t1) and is_soft(t2):
#             store = self.store_getter()
            
#             # Extract inner terms
#             t1_inner = t1.arguments[0]
#             t2_inner = t2.arguments[0]
            
#             try:
#                 # Get soft unification fact
#                 soft_fact = get_unify_fact(t1_inner, t2_inner, store, self.metric)
                
#                 # CRITICAL: Return the soft_fact in the third position of the tuple
#                 # This is what adds it to the proof's soft facts
#                 yield Fact(Expr("type", t1, t2)), {}, {soft_fact}
#             except Exception as e:
#                 print(f"Error in TypeExternal: {e}")
#                 # Still succeed but without soft facts
#                 yield Fact(Expr("type", t1, t2)), {}, set()
#         else:
#             # For non-soft terms, just succeed
#             yield Fact(Expr("type", t1, t2)), {}, set()

# class TypeExternal(External):
#     def __init__(self, store_getter, metric):
#         super().__init__("type", 2, None)
#         self.store_getter = store_getter
#         self.metric = metric
        
#     def get_answers(self, t1, t2) -> Iterable[tuple[Clause, dict, set]]:
#         """Type predicate that directly creates a probabilistic fact"""
#         if is_soft(t1) and is_soft(t2):
#             store = self.store_getter()
            
#             # Extract inner terms
#             t1_inner = t1.arguments[0]
#             t2_inner = t2.arguments[0]
            
#             try:
#                 # Calculate similarity
#                 similarity = store.soft_unify_score(t1_inner, t2_inner, self.metric)
                
#                 # Create type fact
#                 type_fact = Expr("type", t1, t2)
                
#                 # Create probabilistic fact with exact similarity as probability
#                 prob_fact = ProbabilisticFact(similarity, type_fact)
                
#                 # Return this directly - no soft facts needed
#                 yield prob_fact, {}, set()
                
#             except Exception as e:
#                 (f"Error in TypeExternal: {e}")
#                 # Create a low-probability fact on error
#                 yield ProbabilisticFact(0.01, Expr("type", t1, t2)), {}, set()
#         else:
#             # For non-soft terms, just succeed with high probability
#             yield ProbabilisticFact(0.99, Expr("type", t1, t2)), {}, set()


# class TypeExternal(External):
#     def __init__(self, store_getter, metric):
#         super().__init__("type", 2, None)
#         self.store_getter = store_getter
#         self.metric = metric
        
#     def get_answers(self, t1, t2) -> Iterable[tuple[Clause, dict, set]]:
#         if is_soft(t1) and is_soft(t2):
#             store = self.store_getter()
            
#             # Extract inner terms
#             t1_inner = t1.arguments[0]
#             t2_inner = t2.arguments[0]
            
#             try:
#                 # Get log similarity
#                 log_similarity = store.soft_unify_score(t1_inner, t2_inner, self.metric)
                
#                 # Create type fact
#                 type_fact = Expr("type", t1, t2)
                
#                 # Create LogProbabilisticExpr directly with the log probability
#                 log_prob_fact = LogProbabilisticExpr(
#                     log_similarity,  # Use log probability directly
#                     ":-",           # Create a fact
#                     type_fact,      # The type predicate
#                     Expr(","),      # Empty body
#                     infix=True
#                 )
                
#                 yield log_prob_fact, {}, set()
                
#             except Exception as e:
#                 print(f"Error in TypeExternal: {e}")
#                 # Use a small but valid log probability (-4.6 ≈ log(0.01))
#                 yield Fact(LogProbabilisticExpr(-4.6, ":-", Expr("type", t1, t2), Expr(","), infix=True), {}, set())
#         else:
#             # For non-soft terms, use log(0.99) ≈ -0.01
#             yield Fact(LogProbabilisticExpr(-0.01, ":-", Expr("type", t1, t2), Expr(","), infix=True), {}, set())

class TypeExternal(External):
    def __init__(self, store_getter, metric="l2"):
        super().__init__("type", 2, None)
        self.store_getter = store_getter
        self.metric = metric
        self.cache = {}  # Cache for previously computed scores
        
    # def get_answers(self, t1, t2) -> Iterable[tuple[Clause, dict, set]]:
    #     """
    #     Implements type/2 predicate that performs soft unification between terms.
    #     """
    #     print(f"TypeExternal.get_answers called with: {t1}, {t2}")
    #     store = self.store_getter()
        
    #     # Create a cache key for these terms
    #     cache_key = f"{t1}_{t2}"
        
    #     # Check if we've seen this pair before
    #     if cache_key in self.cache:
    #         soft_fact = self.cache[cache_key]
    #         print(f"Returning cached soft_fact: {soft_fact}")
    #         return [(Fact(Expr("type", t1, t2)), {}, {soft_fact})]
        
    #     try:
    #         # Extract inner terms if they're soft terms
    #         t1_inner = t1.arguments[0] if hasattr(t1, 'arguments') and is_soft(t1) else t1
    #         t2_inner = t2.arguments[0] if hasattr(t2, 'arguments') and is_soft(t2) else t2
            
    #         # Default to a high log probability for identical terms
    #         log_score = 0.0  # log(1.0) = 0.0
            
    #         # Calculate soft unification score if terms are different
    #         if t1_inner != t2_inner:
    #             try:
    #                 if hasattr(t1_inner, 'is_tensor') and t1_inner.is_tensor():
    #                     t1_inner = t1_inner.clone()
    #                 if hasattr(t2_inner, 'is_tensor') and t2_inner.is_tensor():
    #                     t2_inner = t2_inner.clone()
    #                 log_score = store.soft_unify_score(t1_inner, t2_inner, self.metric)
    #                 # print(f"Soft unification score (log) between {t1_inner} and {t2_inner}: {log_score}")
    #             except Exception as e:
    #                 print(f"Error calculating soft unification: {e}")
    #                 # Fallback to a low log probability
    #                 log_score = -5.0  # log(0.0067) ≈ -5.0
            
    #         # Create the fact expression
    #         type_expr = Expr("type", t1, t2)
            
    #         # Create a log probabilistic fact using the imported constructor
    #         # We need to create a fact with the structure:
    #         # LogProbabilisticExpr(log_prob, ":-", head, body, infix=True)
    #         # where body is an empty conjunction (representing true)
            
    #         soft_fact = LogProbabilisticExpr(
    #             log_score,  # The log probability
    #             ":-",       # Rule functor
    #             type_expr,  # Head of the rule
    #             Expr(","),  # Empty body (true)
    #             infix=True  # Use infix notation
    #         )
            
    #         # Cache for future use
    #         self.cache[cache_key] = soft_fact
            
    #         # Return the fact expression, empty substitution, and the set with our soft fact
    #         return [(Fact(type_expr), {}, {soft_fact})]
            
    #     except Exception as e:
    #         print(f"Unexpected error in TypeExternal: {e}")
    #         # Return a simple success with very low probability on error
    #         error_fact = LogProbabilisticExpr(-10.0, ":-", Expr("type", t1, t2), Expr(","), infix=True)
    #         return [(Fact(Expr("type", t1, t2)), {}, {error_fact})]
    
    # Debug for TypeExternal.get_answers method
    def get_answers(self, t1, t2) -> Iterable[tuple[Clause, dict, set]]:
        """
        Debug instrumented version to track gradient flow
        """
        # print("\n=== TypeExternal.get_answers called ===")
        # print(f"t1: {t1}, t2: {t2}")
        
        store = self.store_getter()
        
        # Extract inner terms if they're soft terms
        t1_inner = t1.arguments[0] if hasattr(t1, 'arguments') and is_soft(t1) else t1
        t2_inner = t2.arguments[0] if hasattr(t2, 'arguments') and is_soft(t2) else t2
        # print(f"Inner terms: t1_inner={t1_inner}, t2_inner={t2_inner}")
        
        # Default to a high log probability for identical terms
        log_score = 0.0  # log(1.0) = 0.0
        
        # Calculate soft unification score if terms are different
        if t1_inner != t2_inner:
            try:
                # Track embedding retrieval
                # print(f"Getting embeddings for {t1_inner} and {t2_inner}")
                
                # Get the log similarity score
                # print("Calling soft_unify_score...")
                log_score = store.soft_unify_score(t1_inner, t2_inner, self.metric)
                # print(f"Received log_score: {log_score}")
                # print(f"log_score type: {type(log_score)}")
                if isinstance(log_score, torch.Tensor):
                    print(f"log_score requires_grad: {log_score.requires_grad}")
                    
            except Exception as e:
                # print(f"Error calculating soft unification: {e}")
                log_score = -5.0  # Fallback
        
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
        
        print(f"Created soft_fact: {soft_fact}")
        if hasattr(soft_fact, 'get_log_probability'):
            log_prob = soft_fact.get_log_probability()
            # print(f"soft_fact log_probability: {log_prob}")
            # print(f"log_prob type: {type(log_prob)}")
            if isinstance(log_prob, torch.Tensor):
                print(f"log_prob requires_grad: {log_prob.requires_grad}")
        
        # Return the fact expression, empty substitution, and soft fact
        return [(Fact(type_expr), {}, {soft_fact})]