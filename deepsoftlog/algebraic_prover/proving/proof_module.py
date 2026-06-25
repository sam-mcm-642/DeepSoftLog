from collections import defaultdict
from typing import Iterable, Optional

from deepsoftlog.algebraic_prover.builtins import ALL_BUILTINS
from deepsoftlog.algebraic_prover.proving.proof_queue import OrderedProofQueue, ProofQueue
from deepsoftlog.algebraic_prover.proving.proof_tree import ProofTree
from deepsoftlog.algebraic_prover.proving.unify import mgu
from deepsoftlog.algebraic_prover.algebras.boolean_algebra import BOOLEAN_ALGEBRA
from deepsoftlog.algebraic_prover.algebras.sdd2_algebra import DnfAlgebra
from deepsoftlog.algebraic_prover.algebras.probability_algebra import (
    LOG_PROBABILITY_ALGEBRA,
    PROBABILITY_ALGEBRA,
)
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra, Value
from deepsoftlog.algebraic_prover.terms.expression import Clause, Expr
from deepsoftlog.algebraic_prover.terms.variable import CanonicalVariableCounter, fresh_variables
import torch

class ProofModule:
    def __init__(
        self,
        clauses: Iterable[Clause],
        algebra: Algebra,
    ):
        super().__init__()
        self.clauses: set[Clause] = set(clauses)
        self.algebra = algebra
        self.fresh_var_counter = CanonicalVariableCounter(functor="FV_")
        self.queried = None
        self.mask_query = False

    def mgu(self, t1, t2):
        return mgu(t1, t2)
    
    ##3.7 thinking version
    
    def all_matches(self, term: Expr) -> Iterable[tuple[Clause, dict]]:
        predicate = term.get_predicate()
        
        for builtin in self.get_builtins():
            if predicate == builtin.predicate:
                yield from builtin.get_answers(*term.arguments)

        for db_clause in self.clauses:
            
            # Handle both facts and rules correctly
            if isinstance(db_clause, Expr) and db_clause.functor != ':-':  # It's a fact
                db_head = db_clause
            else:  # It's a rule
                if db_clause.functor == ':-':
                    db_head = db_clause.arguments[0]  # Get the head of the rule
                else:
                    db_head = db_clause.arguments[0]
            
                
            if db_head.get_predicate() == predicate:
                if isinstance(db_clause, Expr) and db_clause.functor != ':-':  # For facts
                    fresh_db_clause = db_clause
                    head_to_match = db_head  # No renaming needed for facts
                else:  # For rules
                    fresh_db_clause = self.fresh_variables(db_clause)
                    head_to_match = fresh_db_clause.arguments[0]  # Use the fresh head for matching
                    
                
                result = self.mgu(term, head_to_match)
                
                if result is not None:
                    unifier, new_facts = result
                    if isinstance(fresh_db_clause, Expr):
                        new_clause = fresh_db_clause
                    else:
                        new_clause = fresh_db_clause.apply_substitution(unifier)
                    yield new_clause, unifier, new_facts    
         
                    
        
    #             yield from builtin.get_answers(*term.arguments)

            
    #         # Handle both facts and rules correctly
            
    #             ##AND ALSO STOPS A QUERY FROM BEING USED AS A FACT
    #             ##WHICH IN TURN STOPS THE QUERY FROM BEING PROVEN BY ITSELF OR
    #             #AN IDENITCAL FACT
                    
                
    #             # Get the correct term to match against
                
                
    #                 unifier, new_facts = result
                    
    #                 yield new_clause, unifier, new_facts


    def fresh_variables(self, term: Clause) -> Clause:
        """Replace all variables in a clause with fresh variables"""
        return fresh_variables(term, self.fresh_var_counter.get_fresh_variable)[0]

    def get_builtins(self):
        return ALL_BUILTINS

    #     self,
    #     query: Expr,
    #     max_proofs: Optional[int] = None,
    #     max_depth: Optional[int] = None,
    #     max_branching: Optional[int] = None,
    #     queue: Optional[ProofQueue] = None,
    #     return_stats: bool = False,
    # ):
        
    #     formulas, proof_steps, nb_proofs = get_proofs(
    #         self,
    #     )
    
    def query(
        self,
        query: Expr,
        max_proofs: Optional[int] = None,
        max_depth: Optional[int] = None,
        max_branching: Optional[int] = None,
        queue: Optional[ProofQueue] = None,
        return_stats: bool = False,
    ):
        print(f"QUERY: {query}")
        self.queried = query
        if queue is None:
            queue = OrderedProofQueue(self.algebra)
        
        formulas, proof_steps, nb_proofs = get_proofs(
            self,
            self.algebra,
            query=query,
            max_proofs=max_proofs,
            max_depth=max_depth,
            queue=queue,
            max_branching=max_branching,
        )
        
        
        # Check for empty formulas
        if not formulas:
            print("WARNING: No formulas found in proof search")
        
        # Evaluate formulas
        result = {}
        for k, f in formulas.items():
            eval_result = self.algebra.evaluate(f)
            result[k] = eval_result
            
            # # Check for numerical issues
            
        
        # Filter out zero results
        zero = self.algebra.eval_zero()
        filtered_result = {k: v for k, v in result.items() if v != zero}
        
        if len(result) != len(filtered_result):
            pass
            print(f"Filtered out {len(result) - len(filtered_result)} zero results")
        
        
        if return_stats:
            return filtered_result, proof_steps, nb_proofs
        return filtered_result

    #     result, proof_steps, nb_proofs = self.query(query, return_stats=True, **kwargs)
    #     nb_proofs
    
    #     result, proof_steps, nb_proofs = self.query(query, return_stats=True, **kwargs)
        
    #     # Debug the original query and result keys
        
                
    #             # Additional checks that might help
                
    #             # Compare string representations
                
    #             # Try comparison after normalization
        
        
        

    
    def __call__(self, query: Expr, **kwargs):
        result, proof_steps, nb_proofs = self.query(query, return_stats=True, **kwargs)
        
        # Debug the original query and result keys
        
                
                # # Additional checks that might help
                
                # Compare string representations
                
                # Try comparison after normalization
        
        if type(result) is set:
            return_value = (len(result) > 0.0, proof_steps, nb_proofs)
            return return_value
        
        if type(result) is dict and query in result:
            return_value = result[query]
            return return_value, proof_steps, nb_proofs
        
        # This is the case we're interested in
        zero_value = self.algebra.evaluate(self.algebra.zero())
        return zero_value, proof_steps, nb_proofs
        
    def eval(self):
        self.store = self.store.eval()
        return self

    def apply(self, *args, **kwargs):
        return self.store.apply(*args, **kwargs)

    def modules(self):
        return self.store.modules()


class BooleanProofModule(ProofModule):
    def __init__(self, clauses):
        super().__init__(clauses, algebra=BOOLEAN_ALGEBRA)

    def query(self, *args, **kwargs):
        result = super().query(*args, **kwargs)
        return set(result.keys())


class ProbabilisticProofModule(ProofModule):
    def __init__(self, clauses, log_mode=False):
        eval_algebra = LOG_PROBABILITY_ALGEBRA if log_mode else PROBABILITY_ALGEBRA
        super().__init__(clauses, algebra=DnfAlgebra(eval_algebra))


def get_proofs(prover, algebra, **kwargs) -> tuple[dict[Expr, Value], int, int]:
    proof_tree = ProofTree(prover, algebra=algebra, **kwargs)
    proofs = defaultdict(algebra.zero)
    nb_proofs = 0
    for proof in proof_tree.get_proofs():
        proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
        nb_proofs += 1

    return dict(proofs), proof_tree.nb_steps, nb_proofs

# # Modify the get_proofs function in proof_module.py
    
#         # Ensure the query being used as the key matches the original query
#         # For complex queries with multiple goals/conjunctions
        
#         # Use the original query as the key if available
#             proofs[query_key] = algebra.add(proofs[query_key], proof.value) 
#             # Otherwise use the proof's query
#             proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
            
#         nb_proofs += 1


    
# #     # Create and run proof tree
    
        
# #         proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
# #         nb_proofs += 1
    
    
# #     # Check if no proofs found
        
# #         # Check for incomplete proofs with only object predicates
            
                    
# #                     # Look for soft unifications that should allow completion
    
# #     # Return the results as normal - no artificial fixes

#     """Enhanced get_proofs that extracts bbox_id and soft unification facts"""
    
#         # Extract Y variable binding (contains bbox_id)
        
#                 variable_bindings[var_name] = str(val)
                
#                 # Specifically capture Y variable (bbox_id)
        
#         # Extract soft unification facts
        
#         # Method 1: Extract from proof.value (for SDD2/DnfAlgebra)
#                     'type': 'soft_fact',
#                     'fact': str(fact),
#                     'log_probability': None
#                 }
                
#                 # Try to get the probability of this soft fact
#                         soft_unif_info['log_probability'] = float(log_prob) if hasattr(log_prob, 'item') else float(log_prob)
                
#                 # Try to extract more details about the soft unification
#                         soft_unif_info['term1'] = str(args[0])
#                         soft_unif_info['term2'] = str(args[1])
                
#                 soft_unifications.append(soft_unif_info)
        
#         # Method 2: Extract from proof.value if it's a different algebra type
#             soft_unifications.append({
#                 'type': 'proof_value',
#                 'log_probability': float(proof.value.log_probability) if hasattr(proof.value.log_probability, 'item') else float(proof.value.log_probability)
#             })
        
#         # Method 3: Try to extract from the proof tree's soft unification cache
#             # Look for recent soft unifications that might be related to this proof
#                     soft_unifications.append({
#                         'type': 'cached_soft_unification',
#                         'terms': str(cache_key),
#                         'score': float(cache_value.item()) if hasattr(cache_value, 'item') else float(cache_value)
#                     })
        
#         # Store comprehensive metadata for this proof
#         proof_metadata[proof.query] = {
#             'bbox_id': bbox_id,
#             'variable_bindings': variable_bindings,
#             'soft_unifications': soft_unifications,
#             'proof_depth': getattr(proof, 'depth', 0),
#             'proof_goals': [str(goal) for goal in getattr(proof, 'goals', [])],
#             'proof_value_type': type(proof.value).__name__
#         }
        
#         # Debug output
        
#         proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
#         nb_proofs += 1
    
#     # Store metadata in the prover for later retrieval
#     prover._last_proof_metadata = proof_metadata
    
    

# DEBUGGING VERSION: Enhanced get_proofs with more detailed debugging
# 
import math
def get_proofs(prover, algebra, **kwargs) -> tuple[dict[Expr, Value], int, int]:
    """Enhanced get_proofs that captures variable bindings and soft unifications"""
    proof_tree = ProofTree(prover, algebra=algebra, **kwargs)
    proofs = defaultdict(algebra.zero)
    proof_metadata = {}
    nb_proofs = 0
    
    for proof in proof_tree.get_proofs():
        # Extract variable bindings (handles fresh variables)
        variable_bindings = {}
        target_object = None
        bbox_id = None
        
        if hasattr(proof, 'current_bindings') and proof.current_bindings:
            from deepsoftlog.algebraic_prover.terms.variable import Variable
            
            # Convert all bindings to string format
            for var, val in proof.current_bindings.items():
                var_name = var.name if isinstance(var, Variable) else str(var)
                val_str = str(val)
                variable_bindings[var_name] = val_str
            
            # Follow X variable chain to find target object
            x_fresh_var = variable_bindings.get('X')
            if x_fresh_var and x_fresh_var in variable_bindings:
                target_object = variable_bindings[x_fresh_var]
            
            # Find first bbox_id (this is from target(X) :- object(X, Y))
            for var_name, val in variable_bindings.items():
                if isinstance(val, str) and (val.startswith('bbox') or val.startswith('att')):
                    bbox_id = val
                    break  # Take the first one (from target rule)
        
        # Extract soft unifications from proof.value
        soft_unifications = []
        
        if hasattr(proof.value, 'pos_facts') and proof.value.pos_facts:
            for fact in proof.value.pos_facts:
                soft_unif_info = {
                    'type': 'soft_fact',
                    'fact': str(fact),
                    'log_probability': None,
                    'probability': None
                }
                
                # Parse the soft fact to extract probability and terms
                fact_str = str(fact)
                
                # Extract probability: "0.0021::k(term1,term2)" or "1::type(~x,~y)"
                import re
                prob_match = re.search(r'(\d+\.?\d*)::', fact_str)
                if prob_match:
                    prob_val = float(prob_match.group(1))
                    soft_unif_info['probability'] = prob_val
                    soft_unif_info['log_probability'] = math.log(prob_val) if prob_val > 0 else float('-inf')
                
                # Extract unification terms from k(term1,term2) pattern
                k_match = re.search(r'k\(([^,]+),([^)]+)\)', fact_str)
                if k_match:
                    soft_unif_info['term1'] = k_match.group(1)
                    soft_unif_info['term2'] = k_match.group(2)
                    soft_unif_info['unification'] = f"{k_match.group(1)}~{k_match.group(2)}"
                
                # Extract type unifications: type(~x,~y)
                type_match = re.search(r'type\(~?([^,]+),~?([^)]+)\)', fact_str)
                if type_match:
                    soft_unif_info['type_term1'] = type_match.group(1)
                    soft_unif_info['type_term2'] = type_match.group(2)
                    soft_unif_info['type_unification'] = f"{type_match.group(1)}~{type_match.group(2)}"
                
                soft_unifications.append(soft_unif_info)
        
        # Debug output
        if soft_unifications:
            for i, su in enumerate(soft_unifications[:2]):  # Show first 2
                unif_str = su.get('unification', su.get('type_unification', 'unknown'))
                prob = su.get('probability', 'N/A')
        
        # SIMPLE FIX: Make unique key to prevent overwrites
        if bbox_id:
            unique_key = f"{proof.query}__{bbox_id}"
        else:
            unique_key = f"{proof.query}__{nb_proofs}"  # Use proof counter as fallback

        # Store comprehensive metadata for this proof using the UNIQUE KEY
        proof_metadata[unique_key] = {  # ← Changed from proof.query to unique_key
            'bbox_id': bbox_id,
            'target_object': target_object,
            'variable_bindings': variable_bindings,
            'soft_unifications': soft_unifications,
            'num_soft_unifications': len(soft_unifications),
            'proof_depth': getattr(proof, 'depth', 0),
            'proof_value_type': type(proof.value).__name__
        }


        proofs[unique_key] = algebra.add(proofs[unique_key], proof.value)
        
        proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
        nb_proofs += 1
    
    # Store metadata in the prover for retrieval by evaluator
    prover._last_proof_metadata = proof_metadata
    
    
    return dict(proofs), proof_tree.nb_steps, nb_proofs