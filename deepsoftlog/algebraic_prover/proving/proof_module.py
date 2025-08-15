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
        # print(f"\nDEBUG: Attempting to match term: {term}")
        # print(f"DEBUG: Looking for predicate: {predicate}")
        
        for builtin in self.get_builtins():
            if predicate == builtin.predicate:
                yield from builtin.get_answers(*term.arguments)

        for db_clause in self.clauses:
            # print(f"\nDEBUG: Examining clause: {db_clause}")
            
            # Handle both facts and rules correctly
            if isinstance(db_clause, Expr) and db_clause.functor != ':-':  # It's a fact
                db_head = db_clause
                # print(f"DEBUG: Fact with predicate: {db_head.get_predicate()}")
            else:  # It's a rule
                if db_clause.functor == ':-':
                    db_head = db_clause.arguments[0]  # Get the head of the rule
                else:
                    db_head = db_clause.arguments[0]
                # print(f"DEBUG: Rule with head predicate: {db_head.get_predicate()}")
            
            # if self.mask_query and db_head == self.queried:
            #     continue
                
            if db_head.get_predicate() == predicate:
                # print(f"DEBUG: Found matching predicate")
                if isinstance(db_clause, Expr) and db_clause.functor != ':-':  # For facts
                    fresh_db_clause = db_clause
                    head_to_match = db_head  # No renaming needed for facts
                else:  # For rules
                    fresh_db_clause = self.fresh_variables(db_clause)
                    head_to_match = fresh_db_clause.arguments[0]  # Use the fresh head for matching
                    
                # print(f"DEBUG: Working with clause: {fresh_db_clause}")
                # print(f"DEBUG: Attempting to match with: {head_to_match}")  # Now correctly using head with fresh variables
                
                result = self.mgu(term, head_to_match)
                # print(f"DEBUG: MGU result: {result}")
                
                if result is not None:
                    unifier, new_facts = result
                    # print(f"DEBUG: Unifier: {unifier}")
                    if isinstance(fresh_db_clause, Expr):
                        new_clause = fresh_db_clause
                    else:
                        new_clause = fresh_db_clause.apply_substitution(unifier)
                    # print(f"DEBUG: Yielding match: {new_clause}")
                    yield new_clause, unifier, new_facts    
         
                    
    # def all_matches(self, term: Expr) -> Iterable[tuple[Clause, dict]]:
    #     predicate = term.get_predicate()
    #     print(f"\nDEBUG: Attempting to match term: {term}")
    #     print(f"DEBUG: Looking for predicate: {predicate}")
        
    #     for builtin in self.get_builtins():
    #         if predicate == builtin.predicate:
    #             yield from builtin.get_answers(*term.arguments)

    #     for db_clause in self.clauses:
    #         print(f"\nDEBUG: Examining clause: {db_clause} with functor: {db_clause.functor}")
            
    #         # Handle both facts and rules correctly
    #         if isinstance(db_clause, Expr) and db_clause.functor != ':-':  # It's a fact
    #             db_head = db_clause
    #             print(f"DEBUG: Fact with predicate: {db_head.get_predicate()}")
    #         else:  # It's a rule
    #             if db_clause.functor == ':-':
    #                 db_head = db_clause.arguments[0]  # Get the head of the rule
    #             else:
    #                 db_head = db_clause.arguments[0]
    #             print(f"DEBUG: Rule with head predicate: {db_head.get_predicate()}")
            
    #         if self.mask_query and db_head == self.queried: ###THIS LINE STOPS A QUERY FROM BEING REPEATED
    #             ##AND ALSO STOPS A QUERY FROM BEING USED AS A FACT
    #             ##WHICH IN TURN STOPS THE QUERY FROM BEING PROVEN BY ITSELF OR
    #             #AN IDENITCAL FACT
    #             continue
    #         print(f"DEBUG: Clause head: {db_head.get_predicate()}, looking for: {predicate}")    
    #         if db_head.get_predicate() == predicate:
    #             print(f"DEBUG: Found matching predicate")
    #             if isinstance(db_clause, Expr) and db_clause.functor != ':-':  # For facts
    #                 fresh_db_clause = db_clause
    #             else:  # For rules
    #                 fresh_db_clause = self.fresh_variables(db_clause)
                    
    #             print(f"DEBUG: Working with clause: {fresh_db_clause}")
                
    #             # Get the correct term to match against
    #             head_to_match = db_head if isinstance(fresh_db_clause, Expr) else fresh_db_clause.arguments[0]
    #             print(f"DEBUG: Attempting to match with: {head_to_match}")
                
    #             result = self.mgu(term, head_to_match)
    #             print(f"DEBUG: MGU result: {result}")
                
    #             if result is not None:
    #                 unifier, new_facts = result
    #                 print(f"DEBUG: Unifier: {unifier}")
    #                 if isinstance(fresh_db_clause, Expr):
    #                     new_clause = fresh_db_clause
    #                 else:
    #                     new_clause = fresh_db_clause.apply_substitution(unifier)
    #                 print(f"DEBUG: Yielding match: {new_clause}")
                    
    #                 yield new_clause, unifier, new_facts


    def fresh_variables(self, term: Clause) -> Clause:
        """Replace all variables in a clause with fresh variables"""
        return fresh_variables(term, self.fresh_var_counter.get_fresh_variable)[0]

    def get_builtins(self):
        return ALL_BUILTINS

    # def query(
    #     self,
    #     query: Expr,
    #     max_proofs: Optional[int] = None,
    #     max_depth: Optional[int] = None,
    #     max_branching: Optional[int] = None,
    #     queue: Optional[ProofQueue] = None,
    #     return_stats: bool = False,
    # ):
    #     print("CALLED")
    #     self.queried = query
    #     if queue is None:
    #         queue = OrderedProofQueue(self.algebra)
        
    #     formulas, proof_steps, nb_proofs = get_proofs(
    #         self,
    #         self.algebra,
    #         query=query,
    #         max_proofs=max_proofs,
    #         max_depth=max_depth,
    #         queue=queue,
    #         max_branching=max_branching,
    #     )
    #     print(f"Proof step result: {formulas}, type: {type(formulas)}")
    #     result = {k: self.algebra.evaluate(f) for k, f in formulas.items()}
    #     print(f"Proof step result: {formulas}, type: {type(formulas)}")
    #     zero = self.algebra.eval_zero()
    #     result = {k: v for k, v in result.items() if v != zero}
    #     if return_stats:
    #         return result, proof_steps, nb_proofs
    #     return result
    
    def query(
        self,
        query: Expr,
        max_proofs: Optional[int] = None,
        max_depth: Optional[int] = None,
        max_branching: Optional[int] = None,
        queue: Optional[ProofQueue] = None,
        return_stats: bool = False,
    ):
        # print(f"CALLED (SPL)")
        print(f"QUERY: {query}")
        self.queried = query
        if queue is None:
            queue = OrderedProofQueue(self.algebra)
            # print(f"Created new OrderedProofQueue with algebra: {type(self.algebra).__name__}")
        
        # print(f"Starting proof search with max_proofs={max_proofs}, max_depth={max_depth}, max_branching={max_branching}")
        formulas, proof_steps, nb_proofs = get_proofs(
            self,
            self.algebra,
            query=query,
            max_proofs=max_proofs,
            max_depth=max_depth,
            queue=queue,
            max_branching=max_branching,
        )
        
        #print(f"Raw formulas returned: {formulas}")
        #print(f"Proof steps: {proof_steps}, Number of proofs: {nb_proofs}")
        
        # Check for empty formulas
        if not formulas:
            print("WARNING: No formulas found in proof search")
        
        # Evaluate formulas
        # print("Evaluating formulas with algebra...")
        result = {}
        for k, f in formulas.items():
            eval_result = self.algebra.evaluate(f)
            result[k] = eval_result
            
            # # Check for numerical issues
            # if isinstance(eval_result, torch.Tensor):
            #     if torch.isneginf(eval_result):
            #         print(f"CRITICAL: Formula {k} evaluated to -inf")
            #     elif torch.isnan(eval_result):
            #         print(f"CRITICAL: Formula {k} evaluated to NaN")
            
            #print(f"Formula {k}: {f} -> {eval_result}")
        
        # Filter out zero results
        zero = self.algebra.eval_zero()
        filtered_result = {k: v for k, v in result.items() if v != zero}
        
        if len(result) != len(filtered_result):
            pass
            # print(f"Filtered out {len(result) - len(filtered_result)} zero results")
        
        # print(f"Final result: {filtered_result}")
        
        if return_stats:
            return filtered_result, proof_steps, nb_proofs
        return filtered_result

    # def __call__(self, query: Expr, **kwargs):
    #     result, proof_steps, nb_proofs = self.query(query, return_stats=True, **kwargs)
    #     print(f"__call__ Result: {result}, type: {type(result)}")
    #     if type(result) is set:
    #         return len(result) > 0.0, proof_steps, nb_proofs
    #     if type(result) is dict and query in result:
    #         print(f"key:")
    #         return result[query], proof_steps, nb_proofs
    #     print(f"evaluate: {self.algebra.evaluate(self.algebra.zero())}")
    #     return self.algebra.evaluate(self.algebra.zero()), proof_steps,
    #     nb_proofs
    
    # def __call__(self, query: Expr, **kwargs):
    #     result, proof_steps, nb_proofs = self.query(query, return_stats=True, **kwargs)
    #     print(f"__call__ Result: {result}, type: {type(result)}")
        
    #     # Debug the original query and result keys
    #     print(f"Original query: {query}")
    #     print(f"Original query type: {type(query)}")
        
    #     if isinstance(result, dict) and result:
    #         print("Result keys:")
    #         for i, key in enumerate(result.keys()):
    #             print(f"  Key {i}: {key}")
    #             print(f"  Key {i} type: {type(key)}")
                
    #             # Additional checks that might help
    #             if hasattr(key, 'functor'):
    #                 print(f"  Key {i} functor: {key.functor}")
    #             if hasattr(query, 'functor') and hasattr(key, 'functor'):
    #                 print(f"  Functors match: {query.functor == key.functor}")
                
    #             # Compare string representations
    #             print(f"  String match: {str(key) == str(query)}")
                
    #             # Try comparison after normalization
    #             if hasattr(query, 'normalize') and hasattr(key, 'normalize'):
    #                 print(f"  Normalized match: {query.normalize() == key.normalize()}")
        
    #     if type(result) is set:
    #         return len(result) > 0.0, proof_steps, nb_proofs
        
    #     if type(result) is dict and query in result:
    #         return result[query], proof_steps, nb_proofs
        
    #     print(f"evaluate: {self.algebra.evaluate(self.algebra.zero())}")
    #     return self.algebra.evaluate(self.algebra.zero()), proof_steps, nb_proofs

    
    def __call__(self, query: Expr, **kwargs):
        result, proof_steps, nb_proofs = self.query(query, return_stats=True, **kwargs)
        # print(f"__call__ Result: {result}, type: {type(result)}")
        
        # Debug the original query and result keys
        # print(f"Original query: {query}")
        # print(f"Original query type: {type(query)}")
        
        # if isinstance(result, dict) and result:
        #     # print("Result keys:")
        #     for i, key in enumerate(result.keys()):
                # print(f"  Key {i}: {key}")
                # print(f"  Key {i} type: {type(key)}")
                
                # # Additional checks that might help
                # if hasattr(key, 'functor'):
                #     # print(f"  Key {i} functor: {key.functor}")
                # if hasattr(query, 'functor') and hasattr(key, 'functor'):
                #     print(f"  Functors match: {query.functor == key.functor}")
                
                # Compare string representations
                # print(f"  String match: {str(key) == str(query)}")
                
                # Try comparison after normalization
                # if hasattr(query, 'normalize') and hasattr(key, 'normalize'):
                #     print(f"  Normalized match: {query.normalize() == key.normalize()}")
        
        if type(result) is set:
            return_value = (len(result) > 0.0, proof_steps, nb_proofs)
            # print(f"Returning set result: {return_value[0]}, type: {type(return_value[0])}")
            return return_value
        
        if type(result) is dict and query in result:
            return_value = result[query]
            # print(f"Returning dict result: {return_value}, type: {type(return_value)}")
            return return_value, proof_steps, nb_proofs
        
        # This is the case we're interested in
        zero_value = self.algebra.evaluate(self.algebra.zero())
        # print(f"No match found, returning: {zero_value}, type: {type(zero_value)}")
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
    # print("get_proofs called")
    proof_tree = ProofTree(prover, algebra=algebra, **kwargs)
    proofs = defaultdict(algebra.zero)
    nb_proofs = 0
    for proof in proof_tree.get_proofs():
        proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
        nb_proofs += 1

    # print("ALL PROOFS", {answer: algebra.evaluate(proof) for answer, proof in proofs.items()})
    return dict(proofs), proof_tree.nb_steps, nb_proofs

# # Modify the get_proofs function in proof_module.py
# def get_proofs(prover, algebra, **kwargs) -> tuple[dict[Expr, Value], int, int]:
#     proof_tree = ProofTree(prover, algebra=algebra, **kwargs)
#     proofs = defaultdict(algebra.zero)
#     nb_proofs = 0
    
#     for proof in proof_tree.get_proofs():
#         # Ensure the query being used as the key matches the original query
#         # For complex queries with multiple goals/conjunctions
#         query_key = kwargs.get('query')  # Get the original query
        
#         # Use the original query as the key if available
#         if query_key is not None and proof.query.get_predicate() == query_key.get_predicate():
#             proofs[query_key] = algebra.add(proofs[query_key], proof.value) 
#         else:
#             # Otherwise use the proof's query
#             proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
            
#         nb_proofs += 1

#     return dict(proofs), proof_tree.nb_steps, nb_proofs

# # def get_proofs(prover, algebra, **kwargs) -> tuple[dict[Expr, Value], int, int]:
# #     # print(f"get_proofs called with kwargs: {kwargs}")
    
# #     # Create and run proof tree
# #     proof_tree = ProofTree(prover, algebra=algebra, **kwargs)
# #     proofs = defaultdict(algebra.zero)
# #     nb_proofs = 0
    
# #     for proof in proof_tree.get_proofs():
# #         # print(f"Found proof: {proof.query}")
# #         # print(f"  Goals: {proof.goals}")
# #         # print(f"  Value: {proof.value}")
        
# #         proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
# #         nb_proofs += 1
    
# #     # print(f"Proof collection complete: {nb_proofs} proofs in {proof_tree.nb_steps} steps")
    
# #     # Check if no proofs found
# #     if nb_proofs == 0:
# #         # print("WARNING: No proofs found in proof tree")
# #         # print(f"Proof tree stats:")
# #         # print(f"  Steps: {proof_tree.nb_steps}")
# #         # print(f"  Answers: {proof_tree.answers}")
# #         # print(f"  Value: {proof_tree.value}")
        
# #         # Check for incomplete proofs with only object predicates
# #         if hasattr(proof_tree, '_proof_history'):
# #             near_complete = [p for _, p in proof_tree._proof_history 
# #                            if p.goals and all(g.functor == "object" for g in p.goals)]
            
# #             # if near_complete:
# #             #     # print(f"Found {len(near_complete)} proofs with only object predicates:")
# #             #     for i, p in enumerate(near_complete[:3]):  # Show at most 3
# #                     # print(f"  Near-complete proof {i}:")
# #                     # print(f"    Query: {p.query}")
# #                     # print(f"    Goals: {p.goals}")
# #                     # print(f"    Value: {p.value}")
# #                     # print(f"    is_complete(): {p.is_complete()}")
                    
# #                     # Look for soft unifications that should allow completion
# #                     # if hasattr(p.value, 'pos_facts'):
# #                     #     print(f"    Soft unifications: {p.value.pos_facts}")
    
# #     # Return the results as normal - no artificial fixes
# #     return dict(proofs), proof_tree.nb_steps, nb_proofs

# def get_proofs(prover, algebra, **kwargs) -> tuple[dict[Expr, Value], int, int]:
#     """Enhanced get_proofs that extracts bbox_id and soft unification facts"""
#     proof_tree = ProofTree(prover, algebra=algebra, **kwargs)
#     proofs = defaultdict(algebra.zero)
#     proof_metadata = {}  # Store detailed information for each proof
#     nb_proofs = 0
    
#     for proof in proof_tree.get_proofs():
#         # Extract Y variable binding (contains bbox_id)
#         bbox_id = None
#         variable_bindings = {}
        
#         if hasattr(proof, 'current_bindings') and proof.current_bindings:
#             from deepsoftlog.algebraic_prover.terms.variable import Variable
#             for var, val in proof.current_bindings.items():
#                 var_name = var.name if isinstance(var, Variable) else str(var)
#                 variable_bindings[var_name] = str(val)
                
#                 # Specifically capture Y variable (bbox_id)
#                 if isinstance(var, Variable) and var.name == 'Y':
#                     bbox_id = str(val)
#                     print(f"EXTRACTED Y binding: Y = {bbox_id} for query {proof.query}")
        
#         # Extract soft unification facts
#         soft_unifications = []
        
#         # Method 1: Extract from proof.value (for SDD2/DnfAlgebra)
#         if hasattr(proof.value, 'pos_facts') and proof.value.pos_facts:
#             for fact in proof.value.pos_facts:
#                 soft_unif_info = {
#                     'type': 'soft_fact',
#                     'fact': str(fact),
#                     'log_probability': None
#                 }
                
#                 # Try to get the probability of this soft fact
#                 if hasattr(fact, 'get_log_probability'):
#                     try:
#                         log_prob = fact.get_log_probability()
#                         soft_unif_info['log_probability'] = float(log_prob) if hasattr(log_prob, 'item') else float(log_prob)
#                     except:
#                         pass
                
#                 # Try to extract more details about the soft unification
#                 if hasattr(fact, 'args') or hasattr(fact, 'arguments'):
#                     args = getattr(fact, 'args', getattr(fact, 'arguments', []))
#                     if len(args) >= 2:
#                         soft_unif_info['term1'] = str(args[0])
#                         soft_unif_info['term2'] = str(args[1])
                
#                 soft_unifications.append(soft_unif_info)
        
#         # Method 2: Extract from proof.value if it's a different algebra type
#         elif hasattr(proof.value, 'log_probability'):
#             soft_unifications.append({
#                 'type': 'proof_value',
#                 'log_probability': float(proof.value.log_probability) if hasattr(proof.value.log_probability, 'item') else float(proof.value.log_probability)
#             })
        
#         # Method 3: Try to extract from the proof tree's soft unification cache
#         if hasattr(prover, 'soft_unification_cache') and prover.soft_unification_cache:
#             # Look for recent soft unifications that might be related to this proof
#             for cache_key, cache_value in prover.soft_unification_cache.items():
#                 if hasattr(cache_value, 'item'):
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
#         if bbox_id or soft_unifications:
#             print(f"PROOF METADATA for {proof.query}:")
#             print(f"  bbox_id: {bbox_id}")
#             print(f"  soft_unifications: {len(soft_unifications)} found")
#             if soft_unifications:
#                 for i, su in enumerate(soft_unifications[:3]):  # Show first 3
#                     print(f"    {i}: {su}")
        
#         proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
#         nb_proofs += 1
    
#     # Store metadata in the prover for later retrieval
#     prover._last_proof_metadata = proof_metadata
    
#     print(f"TOTAL METADATA: Collected metadata for {len(proof_metadata)} proofs")
    
#     return dict(proofs), proof_tree.nb_steps, nb_proofs

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
        
        # Store comprehensive metadata for this proof
        proof_metadata[proof.query] = {
            'bbox_id': bbox_id,
            'target_object': target_object,
            'variable_bindings': variable_bindings,
            'soft_unifications': soft_unifications,
            'num_soft_unifications': len(soft_unifications),
            'proof_depth': getattr(proof, 'depth', 0),
            'proof_value_type': type(proof.value).__name__
        }
        
        # Debug output
        # print(f"PROOF METADATA for {proof.query}:")
        # print(f"  bbox_id: {bbox_id}")
        # print(f"  target_object: {target_object}")
        # print(f"  soft_unifications: {len(soft_unifications)} found")
        if soft_unifications:
            for i, su in enumerate(soft_unifications[:2]):  # Show first 2
                unif_str = su.get('unification', su.get('type_unification', 'unknown'))
                prob = su.get('probability', 'N/A')
                # print(f"    {i}: {unif_str} (prob: {prob})")
        
        proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
        nb_proofs += 1
    
    # Store metadata in the prover for retrieval by evaluator
    prover._last_proof_metadata = proof_metadata
    
    # print(f"TOTAL: Collected metadata for {len(proof_metadata)} proofs")
    
    return dict(proofs), proof_tree.nb_steps, nb_proofs