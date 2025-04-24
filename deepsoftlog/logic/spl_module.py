from typing import Iterable

import torch

from ..algebraic_prover.builtins import External, TypeExternal
from ..algebraic_prover.proving.proof_module import ProofModule
from ..algebraic_prover.algebras.sdd2_algebra import DnfAlgebra
from ..algebraic_prover.algebras.probability_algebra import LOG_PROBABILITY_ALGEBRA
from ..algebraic_prover.algebras.tnorm_algebra import LogProductAlgebra, LogGodelAlgebra
from ..algebraic_prover.algebras.sdd_algebra import SddAlgebra
from ..algebraic_prover.terms.expression import Expr, Fact, Clause
from ..embeddings.embedding_store import EmbeddingStore
from ..parser.vocabulary import Vocabulary
from .soft_unify import soft_mgu
from deepsoftlog.data import sg_to_prolog
from deepsoftlog.algebraic_prover.terms.transformations import normalize_clauses
from deepsoftlog.algebraic_prover.proving.proof_queue import OrderedProofQueue

class SoftProofModule(ProofModule):
    def __init__(
            self,
            clauses: Iterable[Expr],
            embedding_metric: str = "l2",
            semantics: str = 'sdd2',
    ):
        super().__init__(clauses=clauses, algebra=None)
        self.store = EmbeddingStore(0, None, Vocabulary())
        type_external = TypeExternal(lambda: self.get_store(), embedding_metric)
        self.builtins = super().get_builtins() + (ExternalCut(), type_external)
        self.embedding_metric = embedding_metric
        self.semantics = semantics
        self.algebra = _get_algebra(self.semantics, self)
        self.soft_unification_cache = {}

    def mgu(self, t1, t2):
        # Pass the cache to soft_mgu
        return soft_mgu(t1, t2, self.get_store(), self.embedding_metric, self.soft_unification_cache)

    # def query(self, *args, **kwargs):
    #     print("CALLED (SPL)")
    #     if self.algebra is None:
    #         self.algebra = _get_algebra(self.semantics, self)
    #     self.algebra.reset()
    #     print(f"Super query: {super().query(*args, **kwargs)}")
    #     return super().query(*args, **kwargs)
    
    def query(self, *args, **kwargs):
        print(f"QUERY CALLED: {args[0] if args else None}")
        print(f"Algebra type: {type(self.algebra).__name__}")
        print(self.store)
        
        # Capture original query before any transformation
        original_query = args[0] if args else None
        
        self.queried = original_query
        print(f"Query kwargs: {kwargs}")
        
        if self.algebra is None:
            self.algebra = _get_algebra(self.semantics, self)
        self.algebra.reset()
        
        # Don't add return_stats if it's already in kwargs
        if 'return_stats' in kwargs:
            result = super().query(*args, **kwargs)
            # Check if it's a tuple containing the stats
            if isinstance(result, tuple) and len(result) == 3:
                result_dict, proof_steps, nb_proofs = result
            else:
                # If not, we're probably in a different code path
                print("NOTICE: Result wasn't unpacked as expected")
                result_dict = result
                proof_steps = -1
                nb_proofs = -1
        else:
            result_dict, proof_steps, nb_proofs = super().query(*args, return_stats=True, **kwargs)
        
        print(f"Query result stats:")
        print(f"  Proof steps: {proof_steps}")
        print(f"  Number of proofs: {nb_proofs}")
        print(f"  Result type: {type(result_dict)}")
        
        if isinstance(result_dict, dict):
            print(f"  Result keys: {list(result_dict.keys())}")
            for k, v in result_dict.items():
                print(f"  {k}: {v}")
        
        # Log whether the query itself was found in the results
        if isinstance(result_dict, dict) and original_query is not None:
            if original_query in result_dict:
                print(f"Query found in results with value: {result_dict[original_query]}")
            else:
                print(f"WARNING: Query not found in results")
                print(f"Available keys: {list(result_dict.keys())}")
        
        # Return the appropriate result
        if 'return_stats' in kwargs and kwargs['return_stats']:
            return result_dict, proof_steps, nb_proofs
        else:
            return result_dict
        
    def get_builtins(self):
        return self.builtins

    def get_vocabulary(self):
        return Vocabulary().add_all(self.clauses)

    def get_store(self):
        if hasattr(self.store, "module"):
            return self.store.module
        return self.store

    def parameters(self):
        yield from self.store.parameters()
        if self.semantics == "neural":
            yield from self.algebra.parameters()

    def grad_norm(self, order=2):
        grads = [p.grad.detach().data.flatten()
                 for p in self.parameters() if p.grad is not None]
        if len(grads) == 0:
            return 0
        grad_norm = torch.linalg.norm(torch.hstack(grads), ord=order)
        return grad_norm
    

    def update_clauses(self, DataInstance):
        # Make a copy of the set for iteration
        clauses_copy = list(self.clauses)  # Convert to list to avoid set iteration issues

        # Filter out the "scene_graph" clauses, considering the rule structure
        filtered_clauses = []
        
        # Define exceptions - rules to keep despite having filtered functors
        exceptions = ["~rr1"]  # Add any other special relations here
        
        for expr in clauses_copy:
            if expr.functor == ":-":  # It's a rule or fact-as-rule
                head = expr.arguments[0]  # Get the head of the rule
                
                # Check if this is either not a filtered functor OR it's in our exceptions
                if (head.functor not in ["scene_graph", "object", "groundtruth"] or
                    (hasattr(head, "arguments") and len(head.arguments) > 0 and 
                    str(head.arguments[0]) in exceptions)):
                    filtered_clauses.append(expr)
                    
            else:  # It's a simple expression (not a rule)
                if expr.functor not in ["scene_graph", "object", "groundtruth"]:
                    filtered_clauses.append(expr)

        # Now you can update the clauses with the filtered list
        self.clauses.clear()  # Clear the existing clauses
        self.clauses.update(set(filtered_clauses))  # Add the filtered clauses

        # Optionally, update the store and vocabulary
        sg_clauses = sg_to_prolog(DataInstance)
        sg_clauses = normalize_clauses(sg_clauses)        

        self.clauses.update(sg_clauses)  # Add new clauses from the current instance
        updated_vocabulary = Vocabulary().add_all(self.clauses)
        
        # Update the store's vocabulary
        self.get_store().vocabulary = updated_vocabulary
        
        # Initialize embeddings for new constants
        for constant in updated_vocabulary.get_constants():
            if constant not in self.get_store().constant_embeddings:
                print(f"Initializing new embedding for unseen constant: {constant}")
                self.get_store().constant_embeddings[constant] = self.get_store().initializer(constant)
        
        # Clear the cache to ensure fresh computations
        self.get_store().clear_cache()
        
        print(f"Updated clauses and constant embeddings successfully")
            
    def analyze_failed_proof(self, query: Expr):
        """Special function to analyze why a proof is failing"""
        print(f"\n=== PROOF ANALYSIS for query: {query} ===\n")
        
        # Try with extremely high depth and branching limits
        print("Attempting proof with very high limits...")
        
        # Use OrderedProofQueue to prioritize promising proofs
        queue = OrderedProofQueue(self.algebra)
        
        # First try normal query with high limits
        result, steps, nb_proofs = self.query(
            query, 
            max_depth=20,  # Very high depth
            max_branching=20,  # Very high branching
            queue=queue,
            return_stats=True
        )
        
        print(f"Analysis results: {nb_proofs} proofs in {steps} steps")
        
        if nb_proofs == 0:
            print("\nNo proofs found, analyzing proof tree...")
            
            # Try to analyze the proof tree directly
            if hasattr(queue, '_queue'):
                print(f"Queue still has {len(queue._queue)} items")
                
                # Check the first few items
                for i, (_, _, proof) in enumerate(queue._queue[:3]):
                    print(f"\nQueue item {i}:")
                    print(f"  Query: {proof.query}")
                    print(f"  Goals: {proof.goals}")
                    print(f"  Is complete: {proof.is_complete()}")
                    print(f"  Value: {proof.value}")
                    
                    # Check for object predicates
                    if proof.goals and all(g.functor == "object" for g in proof.goals):
                        print(f"  POSSIBLE ISSUE: All goals are object predicates but proof not complete")
                        print(f"  Goals: {proof.goals}")
                        
                        # Look at the needed soft unifications
                        if hasattr(proof.value, 'pos_facts'):
                            print(f"  Soft facts: {proof.value.pos_facts}")
                        
                        # Try to manually complete this proof
                        print(f"  Attempting manual completion...")
                        try:
                            manual_proof = type(proof)(
                                query=proof.query,
                                goals=tuple(),  # Empty tuple = completed
                                depth=proof.depth + 1,
                                proof_tree=proof.proof_tree,
                                value=proof.value
                            )
                            print(f"  Manual proof is_complete: {manual_proof.is_complete()}")
                            
                            # Check if this would be a valid completion
                            if manual_proof.is_complete():
                                print(f"  DIAGNOSTIC: The proof CAN be manually completed!")
                                print(f"  This suggests the issue is in how proof completion is detected")
                        except Exception as e:
                            print(f"  Error creating manual proof: {str(e)}")
        
        print("\n=== END PROOF ANALYSIS ===\n")
        return result


def _get_algebra(semantics, program):
    
    if semantics == "sdd":
        return SddAlgebra(LOG_PROBABILITY_ALGEBRA)
    elif semantics == "sdd2":
        return DnfAlgebra(LOG_PROBABILITY_ALGEBRA)
    elif semantics == "godel":
        return LogGodelAlgebra()
    elif semantics == "product":
        return LogProductAlgebra()
    raise ValueError(f"Unknown semantics: {semantics}")


class ExternalCut(External):
    def __init__(self):
        super().__init__("cut", 1, None)
        self.cache = set()

    def get_answers(self, t1) -> Iterable[tuple[Expr, dict, set]]:
        if t1 not in self.cache:
            self.cache.add(t1)
            fact = Fact(Expr("cut", t1))
            return [(fact, {}, set())]
        return []


class DebugSoftProofModule(SoftProofModule):
    pass
    def __init__(self, clauses: Iterable[Expr], embedding_metric: str = "l2", semantics: str = 'sdd2'):
        super().__init__(clauses=clauses, embedding_metric=embedding_metric, semantics=semantics)
        self.debug = True
        
    def all_matches(self, term: Expr) -> Iterable[tuple[Clause, dict, set]]:
        predicate = term.get_predicate()
        if self.debug:
            print(f"\nDEBUG: Matching term: {term}")
            print(f"DEBUG: Predicate: {predicate}")
        
        # First handle builtins
        for builtin in self.get_builtins():
            if predicate == builtin.predicate:
                if self.debug:
                    print(f"DEBUG: Found builtin match: {builtin.predicate}")
                yield from builtin.get_answers(*term.arguments)

        # Debug all available clauses
        if self.debug:
            print("\nDEBUG: Available clauses:")
            for clause in self.clauses:
                print(f"DEBUG: {clause}")

        # Try to match with all clauses
        for db_clause in self.clauses:
            if self.debug:
                print(f"\nDEBUG: Trying clause: {db_clause}")
            
            # Handle facts (clauses without body)
            is_fact = isinstance(db_clause, Expr) or (
                hasattr(db_clause, 'arguments') and 
                len(db_clause.arguments) == 1
            )
            
            db_head = db_clause if is_fact else db_clause.arguments[0]
            
            if self.debug:
                print(f"DEBUG: Clause head: {db_head}")
                print(f"DEBUG: Is fact: {is_fact}")
            
            if self.mask_query and db_head == self.queried:
                if self.debug:
                    print("DEBUG: Skipping masked query")
                continue
            
            if db_head.get_predicate() == predicate:
                if self.debug:
                    print("DEBUG: Predicate matches")
                
                # For facts, use them directly; for rules, create fresh variables
                working_clause = db_clause if is_fact else self.fresh_variables(db_clause)
                working_head = working_clause if is_fact else working_clause.arguments[0]
                
                if self.debug:
                    print(f"DEBUG: Attempting unification between:")
                    print(f"DEBUG:   Term: {term}")
                    print(f"DEBUG:   Head: {working_head}")
                
                result = self.mgu(term, working_head)
                
                if self.debug:
                    print(f"DEBUG: MGU result: {result}")
                
                if result is not None:
                    unifier, new_facts = result
                    if is_fact:
                        # For facts, create a simple clause
                        new_clause = Clause(working_head, None)  # Adjust based on your Clause implementation
                    else:
                        new_clause = working_clause.apply_substitution(unifier)
                    
                    if self.debug:
                        print(f"DEBUG: Match found!")
                        print(f"DEBUG: Unifier: {unifier}")
                        print(f"DEBUG: New clause: {new_clause}")
                    
                    yield new_clause, unifier, new_facts

    def query(self, *args, **kwargs):
        if self.debug:
            print("\nDEBUG: Starting new query")
            print(f"DEBUG: Args: {args}")
            print(f"DEBUG: Kwargs: {kwargs}")
        return super().query(*args, **kwargs)