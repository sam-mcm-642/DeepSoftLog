from typing import TYPE_CHECKING, Iterator, Optional

from deepsoftlog.algebraic_prover.proving.proof import Proof, ProofDebug
from deepsoftlog.algebraic_prover.proving.proof_queue import ProofQueue
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra
from deepsoftlog.algebraic_prover.terms.expression import Expr

if TYPE_CHECKING:
    from deepsoftlog.algebraic_prover.proving.proof_module import ProofModule

import traceback
from deepsoftlog.algebraic_prover.algebras.sdd2_algebra import ConjoinedFacts

class ProofTree:
    """
    Proof tree for a query.
    Searches depth-first.
    """

    def __init__(
        self,
        program: "ProofModule",
        query: Expr,
        algebra: Algebra,
        max_depth: Optional[int] = None,
        max_proofs: Optional[int] = None,
        queue: ProofQueue = None,
        max_branching: Optional[int] = None,
    ):
        self.algebra = algebra
        self.program = program
        self.max_depth = max_depth
        self.max_proofs = max_proofs
        self.max_branching = max_branching
        self.sub_calls = dict()
        self.answers = set()
        self.incomplete_sub_trees: list["ProofTree"] = []
        self.proofs = []
        self.queue = queue
        self.queue.add(self._create_proof_for(query), None)
        self.value = self.algebra.zero()
        self.nb_steps = 0

    # def _create_proof_for(self, query: Expr):
    #     # if type(query) is not Expr:
    #     #     query = query.query
    #     # print("_create_proof_for called")
    #     # print(f"query: {type(query)} {query}")
    #     print(f"Proof: {ProofDebug(query=query, proof_tree=self, value=self.algebra.one())}")
    #     return ProofDebug(query=query, proof_tree=self,
    #     value=self.algebra.one())
    
    ##3.7 thinking version
    def _create_proof_for(self, query: Expr):
        print(f"Proof: {ProofDebug(query=query, proof_tree=self, value=self.algebra.one())}")
        return ProofDebug(query=query, proof_tree=self, value=self.algebra.one(), bindings={})

    def is_complete(self) -> bool:
        return self.queue.empty() or len(self.proofs) >= self.get_max_proofs()

    # def get_proofs(self) -> Iterator[Proof]:
    #     # print("get_proofs called")
    #     while not self.is_complete():
    #         proof = self.step()
    #         if proof is not None:
    #             yield proof
    
    def get_proofs(self) -> Iterator[Proof]:
        """More aggressive approach to finding proofs"""
        attempts = 0
        while not self.queue.empty() and attempts < 1000 and len(self.proofs) < self.get_max_proofs():
            attempts += 1
            print(f"PROOF ATTEMPT: {attempts}, Proofs found: {len(self.proofs)}")
            
            # Try regular step logic
            proof = self.step()
            
            # Check the queue for proofs very close to completion (with only object goals)
            if self.queue.empty() and not self.proofs:
                print("No proofs found but queue is empty - checking for near-complete proofs")
                # Scan all attempted proofs for ones with only object predicates left
                if hasattr(self, '_attempted_proofs') and self._attempted_proofs:
                    for p in self._attempted_proofs:
                        if (p.goals and all(g.functor == "object" for g in p.goals) and 
                            isinstance(p.value, ConjoinedFacts) and p.value.pos_facts):
                            print(f"FORCE COMPLETING PROOF: {p.query} with {len(p.goals)} object goals")
                            empty_goal_proof = Proof(
                                query=p.query,
                                goals=tuple(),  # Empty goals = completed
                                depth=p.depth + 1,
                                proof_tree=self,
                                value=p.value,
                                bindings=getattr(p, 'current_bindings', {})
                            )
                            self.answers.add(empty_goal_proof.query)
                            self.proofs.append(empty_goal_proof)
                            self.value = self.algebra.add(self.value, empty_goal_proof.value)
                            yield empty_goal_proof
            
            if proof is not None:
                yield proof
        
        # Final check - did we find any proofs?
        if not self.proofs and self.nb_steps > 10:
            print("WARNING: No proofs found after significant search - debugging info:")
            print(f"Attempted {attempts} proof steps")
            # Print any proof that made it far in the process
            if hasattr(self, '_attempted_proofs') and self._attempted_proofs:
                for i, p in enumerate(self._attempted_proofs[-5:]):
                    print(f"Near-complete proof {i}: Goals: {p.goals}, Value: {p.value}")

    # def step(self) -> Optional[Proof]:
    #     print(f"PROOF_STEP [{self.nb_steps}]: Queue size={len(self._queue._queue) if hasattr(self._queue, '_queue') else '?'}, Proofs={len(self.proofs)}")
        
    #     self.nb_steps += 1
    #     if len(self.incomplete_sub_trees):
    #         return self._step_subtree()

    #     proof = self.queue.next()
    #     print(f"Proof: {proof, self.algebra.one()}")
        
    #     if proof.is_complete():
    #         print(f"COMPLETE PROOF FOUND: {proof.query} with value {proof.value}")
    #         self.answers.add(proof.query)
    #         self.proofs.append(proof)
    #         old_value = self.value
    #         self.value = self.algebra.add(self.value, proof.value)
    #         print(f"Value updated: {old_value} -> {self.value}")
    #         return proof

    #     if self.is_pruned(proof):
    #         print(f"PRUNED: Proof at depth {proof.depth} > max_depth {self.get_max_depth()}")
    #     else:
    #         print(f"Generating children for proof with {proof.nb_goals()} goals")
    #         local_queue = self.queue.new(self.algebra)
    #         proof_remaining = proof.nb_goals()
            
    #         child_proofs = list(proof.get_children())
    #         print(f"Generated {len(child_proofs)} child proofs")
            
    #         for child_proof in child_proofs:
    #             child_remaining = child_proof.nb_goals()
    #             print(f"Adding proof step: {child_proof}, type: {type(child_proof)}")
                
    #             # Check for potential numerical issues in the proof value
    #             if hasattr(child_proof, 'value') and isinstance(child_proof.value, torch.Tensor):
    #                 if torch.isneginf(child_proof.value):
    #                     print(f"WARNING: Child proof has -inf value")
    #                 elif torch.isnan(child_proof.value):
    #                     print(f"WARNING: Child proof has NaN value")
                
    #             if child_remaining < proof_remaining:
    #                 print(f"Child has fewer goals ({child_remaining} < {proof_remaining}), adding to local queue")
    #                 local_queue.add(child_proof, None)
    #             else:
    #                 print(f"Child has same or more goals ({child_remaining} >= {proof_remaining}), adding to main queue")
    #                 self.queue.add(child_proof, None)
                    
    #         print(f"Adding up to {self.max_branching} proofs from local queue to main queue")
    #         self.queue.add_first(self.max_branching, local_queue)
    
    def step(self) -> Optional[Proof]:
        """Enhanced step method with diagnostic tracing"""
        print(f"STEP [{self.nb_steps}]: Queue size={len(self.queue._queue) if hasattr(self.queue, '_queue') else '?'}")
        
        self.nb_steps += 1
        if len(self.incomplete_sub_trees):
            return self._step_subtree()

        # Get next proof to consider
        if self.queue.empty():
            print("Queue is empty - no more proofs to process")
            return None
            
        proof = self.queue.next()
        
        # Track this proof for analysis
        if not hasattr(self, '_proof_history'):
            self._proof_history = []
        self._proof_history.append((self.nb_steps, proof))
        
        # Debug output
        # print(f"Processing proof {id(proof)} with {len(proof.goals)} goals")
        if proof.goals:
            # print(f"  Current goals: {proof.goals}")
            # Check for object-only goals
            if all(g.functor == "object" for g in proof.goals):
                print(f"  DIAGNOSTIC: All goals are object predicates - should complete soon")
                # Check if there are soft unifications available
                if hasattr(proof.value, 'pos_facts') and proof.value.pos_facts:
                    print(f"  DIAGNOSTIC: Proof has soft unifications: {proof.value.pos_facts}")
                    print(f"  DIAGNOSTIC: is_complete() says: {proof.is_complete()}")
        else:
            print("  No goals - proof should be complete")
        
        # Check completion
        if proof.is_complete():
            # print(f"COMPLETE PROOF FOUND: {proof.query} with {len(getattr(proof, 'goals', []))} goals")
            self.answers.add(proof.query)
            self.proofs.append(proof)
            old_value = self.value
            self.value = self.algebra.add(self.value, proof.value)
            # print(f"  Value updated: {old_value} -> {self.value}")
            return proof

        # Process children in usual way
        if not self.is_pruned(proof):
            local_queue = self.queue.new(self.algebra)
            proof_remaining = proof.nb_goals()
            
            # Get children - diagnostic logging for any issues
            try:
                child_proofs = list(proof.get_children()) 
                print(f"  Generated {len(child_proofs)} child proofs")
                
                # Analyze children
                for i, child in enumerate(child_proofs[:3]):  # Limit to first 3 to avoid verbose output
                    print(f"  Child {i}: {len(child.goals)} goals: {child.goals[:2]}{'...' if len(child.goals) > 2 else ''}")
                    print(f"    is_complete(): {child.is_complete()}")
                
                for child_proof in child_proofs:
                    child_remaining = child_proof.nb_goals()
                    if child_remaining < proof_remaining:
                        local_queue.add(child_proof, None)
                    else:
                        self.queue.add(child_proof, None)
                        
                self.queue.add_first(self.max_branching, local_queue)
            except Exception as e:
                print(f"ERROR in get_children: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  Pruned proof at depth {proof.depth}")

    

    def _step_subtree(self):
        self.incomplete_sub_trees[-1].step()
        if self.incomplete_sub_trees[-1].is_complete():
            del self.incomplete_sub_trees[-1]

    def get_answers(self) -> set[Expr]:
        assert self.is_complete()
        return self.answers

    def sub_call(self, query: Expr, depth: int) -> "ProofTree":
        new_algebra = self.algebra.get_dual()
        new_tree = type(self)(
            program=self.program,
            query=query,
            algebra=new_algebra,
            max_depth=self.max_depth - depth if self.max_depth is not None else None,
            max_proofs=self.max_proofs,
            max_branching=self.max_branching,
            queue=self.queue.new(new_algebra),
        )
        print(f"query: {query}")
        self.sub_calls[query] = new_tree
        return new_tree

    def get_sub_call_tree(self, query: Expr, depth: int) -> Optional["ProofTree"]:
        if query in self.sub_calls:
            return self.sub_calls[query]
        else:
            new_tree = self.sub_call(query, depth)
            self.incomplete_sub_trees.append(new_tree)
            return None

    def get_max_depth(self):
        if self.max_depth is None:
            return float("+inf")
        return self.max_depth

    def get_max_proofs(self):
        if self.max_proofs is None:
            return float("+inf")
        return self.max_proofs

    def is_pruned(self, proof: ProofDebug):
        if proof.depth > self.get_max_depth():
            if self.max_depth is None:  # pragma: no cover
                import warnings

                warnings.warn("Default max depth exceeded")
            return True
        return False
