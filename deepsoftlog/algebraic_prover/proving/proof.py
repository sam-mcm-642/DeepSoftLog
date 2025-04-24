from typing import TYPE_CHECKING, Iterable, Optional

from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra, Value
from deepsoftlog.algebraic_prover.terms.expression import Fact, Expr

if TYPE_CHECKING:  # pragma: no cover
    from .proof_tree import ProofTree
import torch


class Proof:
    def __init__(
        self,
        query: Expr,
        goals: Optional[tuple[Expr, ...]] = None,
        depth: int = 0,
        proof_tree: "ProofTree" = None,
        value: Value = None,
        bindings: dict = None,  # Add bindings parameter
    ):
        if goals is None:
            goals = (query,)
        self.query = query
        self.depth = depth
        self.goals: tuple[Expr, ...] = goals
        self.value = value
        self.proof_tree = proof_tree
        self.current_bindings = bindings or {}  # Track accumulated bindings
        
        print("Depth: ",  depth, "new proof", self)

    # def is_complete(self) -> bool:
    #     return len(self.goals) == 0
    def is_complete(self) -> bool:
        """Enhanced is_complete with diagnostic information"""
        # Check for empty goals
        is_goals_empty = len(self.goals) == 0
        
        # Diagnostic info
        if hasattr(self, 'query'):
            print(f"is_complete check for proof with query {self.query}:")
            print(f"  Goals empty: {is_goals_empty}")
            print(f"  Goals: {self.goals}")
            print(f"  Value type: {type(self.value)}")
            
            # Check special case of object-only goals
            if self.goals and all(g.functor == "object" for g in self.goals):
                print(f"  SPECIAL CASE: All goals are object predicates")
                if hasattr(self.value, 'pos_facts'):
                    print(f"  Value has soft facts: {bool(self.value.pos_facts)}")
                    print(f"  Soft facts: {self.value.pos_facts}")
        
        return is_goals_empty



    def nb_goals(self) -> int:
        
        return len(self.goals)

    def get_algebra(self) -> Algebra:
        return self.proof_tree.algebra

    def get_children(self) -> Iterable["Proof"]:
        print(f"goals[0]: {self.goals[0]}")
        if self.goals[0].functor == "\\+":
            yield from self.negation_node()
        else:
            yield from self.apply_clauses()
        # print(f"goals: {self.goals}")

    def negation_node(self):
        # print("negation_node")
        negated_goal = self.goals[0].arguments[0]
        matches = self.proof_tree.program.all_matches(negated_goal)
        # print(f"New goals before substitution: {self.goals}, Types: {[type(g) for g in self.goals]}")
        if not any(matches):
            # goal is not present, so negation is trivially true
            yield self.get_child(new_goals=self.goals[1:])
        else:
            # create proof tree for negation
            sub_call_tree = self.proof_tree.get_sub_call_tree(negated_goal, self.depth)
            if sub_call_tree is None:
                yield self
            else:
                sub_call_value = sub_call_tree.value
                new_value = self.get_algebra().multiply(self.value, sub_call_value)
                yield self.get_child(new_goals=self.goals[1:], value=new_value)

    # def apply_clauses(self):
    #     print("apply_clauses")
    #     first_goal, *remaining = self.goals
    #     print(f"Trying to match goal: {first_goal}")
    #     matches = self.proof_tree.program.all_matches(first_goal)
    #     print(f"Found matches: {list(matches)}")
    #     print(f"Clause: {matches[0][0]}")
        
    #     for clause, unifier, new_facts in matches:
    #         print(f"Matching clause: {clause}")
    #         print(f"With unifier: {unifier}")
            
    #         # For rules, handle the body correctly based on its structure
    #         if clause.is_clause():  # Check if it's a rule
    #             # Extract body properly, handling the structure of conjunctive goals
    #             body = clause.arguments[1]
    #             print(f"Body: {body}")
    #             print(body.arguments)
    #             if body.is_and():  # It's a conjunctive goal with ","
    #                 new_goals = body.arguments  # Get all conjuncts
    #             else:
    #                 new_goals = (body,)  # Single goal
                
    #             # Apply the unifier to each goal in the body
    #             new_goals = tuple(g.apply_substitution(unifier) for g in new_goals)
                
    #             # Add remaining goals with unifier applied
    #             new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
                
    #             query = self.query.apply_substitution(unifier)
    #             new_value = self.create_new_value(clause, new_facts)
                
    #             yield self.get_child(
    #                 query=query,
    #                 new_goals=new_goals,
    #                 depth=self.depth + 1,
    #                 value=new_value,
    #             )
    #         else:  # It's a fact
    #             # Just continue with remaining goals
    #             query = self.query.apply_substitution(unifier)
    #             new_value = self.create_new_value(clause, new_facts)
                
    #             yield self.get_child(
    #                 query=query,
    #                 new_goals=tuple(g.apply_substitution(unifier) for g in remaining),
    #                 depth=self.depth + 1,
    #                 value=new_value,
    #             )
    # def apply_clauses(self):
    #     first_goal, *remaining = self.goals
    #     matches = self.proof_tree.program.all_matches(first_goal)
    #     for clause, unifier, new_facts in matches:
    #         new_goals = clause.arguments[1].arguments  # new goals from clause body
    #         new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
    #         query: Expr = self.query.apply_substitution(unifier)
    #         new_value = self.create_new_value(clause, new_facts)
    #         yield self.get_child(
    #             query=query,
    #             new_goals=new_goals,
    #             depth=self.depth + 1,
    #             value=new_value,
    #         )
    
    # def apply_clauses(self):
    #     if not self.goals:
    #         print("COMPLETE: No more goals to process")
    #         return  # Return an empty iterator
    #     first_goal, *remaining = self.goals
        
    #     # Special handling for conjunctions
    #     if first_goal.functor == "," and first_goal.get_arity() > 0:
    #         # Break down the conjunction into separate goals
    #         conjoined_goals = first_goal.arguments
    #         new_goals = conjoined_goals + tuple(remaining)
    #         # Create a child proof with the broken-down goals
    #         yield self.get_child(
    #             new_goals=new_goals,
    #             depth=self.depth + 1,
    #             value=self.value
    #         )
    #         return
        
    #     # Original code for non-conjunctive goals
    #     matches = self.proof_tree.program.all_matches(first_goal)
    #     for clause, unifier, new_facts in matches:
    #         new_goals = clause.arguments[1].arguments  # new goals from clause body
    #         new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
    #         query: Expr = self.query.apply_substitution(unifier)
    #         new_value = self.create_new_value(clause, new_facts)
    #         yield self.get_child(
    #             query=query,
    #             new_goals=new_goals,
    #             depth=self.depth + 1,
    #             value=new_value,
    #         )
    
    def apply_clauses(self):
        """Enhanced apply_clauses with diagnostic tracing"""
        print(f"apply_clauses called for proof {id(self)} with {len(self.goals)} goals")
        
        # Diagnostic for empty goals
        if not self.goals:
            print("DIAGNOSTIC: apply_clauses called with empty goals")
            yield self  # Simply return this proof as it's already complete
            return
        
        # Diagnostic for object-only goals
        if all(g.functor == "object" for g in self.goals):
            print(f"DIAGNOSTIC: All remaining goals are object predicates: {self.goals}")
            print(f"DIAGNOSTIC: Value has soft unifications: {hasattr(self.value, 'pos_facts')}")
            if hasattr(self.value, 'pos_facts'):
                print(f"DIAGNOSTIC: Soft unifications: {self.value.pos_facts}")
                
                # This is where we need to understand why these aren't being completed
                print(f"DIAGNOSTIC: Trace of goals, value and completion status:")
                print(f"  Goals: {self.goals}")
                print(f"  Value: {self.value}")
                print(f"  is_complete(): {self.is_complete()}")
                if hasattr(self, 'get_child'):
                    # Generate a child with empty goals for analysis
                    test_child = self.get_child(new_goals=tuple())
                    print(f"  Test child with empty goals - is_complete(): {test_child.is_complete()}")
        
        # Standard processing
        first_goal, *remaining = self.goals
        
        # Special handling for conjunctions
        if first_goal.functor == "," and first_goal.get_arity() > 0:
            print(f"Breaking down conjunction: {first_goal}")
            conjoined_goals = first_goal.arguments
            new_goals = conjoined_goals + tuple(remaining)
            new_child = self.get_child(new_goals=new_goals, depth=self.depth + 1, value=self.value)
            print(f"Yielding conjunction breakdown child with {len(new_child.goals)} goals")
            yield new_child
            return
        
        # Process matches
        print(f"Looking for matches for: {first_goal}")
        matches = list(self.proof_tree.program.all_matches(first_goal))
        print(f"Found {len(matches)} matches")
        
        for i, (clause, unifier, new_facts) in enumerate(matches):
            print(f"Processing match {i}: {clause}")
            
            if clause.is_fact():
                print(f"Match is a fact")
                # Fact match - continue with remaining goals
                new_goals = tuple(g.apply_substitution(unifier) for g in remaining)
                query = self.query.apply_substitution(unifier)
                new_value = self.create_new_value(clause, new_facts)
                
                # Diagnostic for object-only remaining goals
                if new_goals and all(g.functor == "object" for g in new_goals):
                    print(f"DIAGNOSTIC: After matching fact, only object goals remain: {new_goals}")
                    print(f"DIAGNOSTIC: New value: {new_value}")
                    print(f"DIAGNOSTIC: New value has soft facts: {hasattr(new_value, 'pos_facts')}")
                    if hasattr(new_value, 'pos_facts'):
                        print(f"DIAGNOSTIC: New soft facts: {new_value.pos_facts}")
                
                child = self.get_child(query=query, new_goals=new_goals, depth=self.depth+1, value=new_value)
                print(f"Yielding fact-matched child with {len(child.goals)} goals")
                print(f"  is_complete(): {child.is_complete()}")
                yield child
            else:
                print(f"Match is a rule")
                # Rule match - add body goals first
                body = clause.arguments[1]
                if body.is_and():
                    new_body_goals = body.arguments
                else:
                    new_body_goals = (body,)
                
                # Apply unifier
                new_body_goals = tuple(g.apply_substitution(unifier) for g in new_body_goals)
                new_remaining = tuple(g.apply_substitution(unifier) for g in remaining)
                new_goals = new_body_goals + new_remaining
                
                query = self.query.apply_substitution(unifier)
                new_value = self.create_new_value(clause, new_facts)
                
                child = self.get_child(query=query, new_goals=new_goals, depth=self.depth+1, value=new_value)
                print(f"Yielding rule-matched child with {len(child.goals)} goals")
                yield child


    # def create_new_value(self, clause, new_facts):
    #     new_facts = self.get_algebra().reduce_mul_value_pos(new_facts)
    #     new_value = self.get_algebra().multiply(self.value, new_facts)
    #     if clause.is_annotated():
    #         new_value = self.get_algebra().multiply_value_pos(new_value, clause)
    #     return new_value
    
    def create_new_value(self, clause, new_facts):
        print(f"Creating new value from clause: {clause}")
        if new_facts:
            print(f"With new_facts: {new_facts}")
        
        new_facts_value = self.get_algebra().reduce_mul_value_pos(new_facts)
        print(f"New facts value: {new_facts_value}")
        
        new_value = self.get_algebra().multiply(self.value, new_facts_value)
        print(f"After multiply with current value: {new_value}")
        
        if clause.is_annotated():
            clause_value = self.get_algebra().value_pos(clause)
            print(f"Clause is annotated with value: {clause_value}")
            new_value = self.get_algebra().multiply_value_pos(new_value, clause)
            print(f"Final value after clause annotation: {new_value}")
        
        # Check for numerical issues
        if isinstance(new_value, torch.Tensor):
            if torch.isneginf(new_value):
                print("CRITICAL: create_new_value produced -inf")
            elif torch.isnan(new_value):
                print("CRITICAL: create_new_value produced NaN")
        
        return new_value

    def get_child(
        self,
        query: Optional[Expr] = None,
        new_goals: tuple[Expr, ...] = tuple(),
        depth: Optional[int] = None,
        value: Optional[Value] = None,
        bindings: Optional[dict] = None,  # Add bindings parameter
    ):
        if new_goals is None:
            new_goals = tuple()
        # print(f"new_goals: {new_goals}")
        # print(self.proof_tree)
        return ProofDebug(
            query=self.query if query is None else query,
            value=self.value if value is None else value,
            goals=new_goals,
            depth=self.depth if depth is None else depth,
            proof_tree=self.proof_tree
        )
        

    def __repr__(self):  # pragma: no cover
        return f"{self.query}: {self.goals} - {self.value}"

    def __lt__(self, other: "Proof"):
        return len(self.goals) < len(other.goals)


class ProofDebug(Proof):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_level = 0
    
    # def apply_clauses(self):
    #     first_goal, *remaining = self.goals
    #     matches = self.proof_tree.program.all_matches(first_goal)
    #     for clause, unifier, new_facts in matches:
    #         new_goals = clause.arguments[1].arguments  # new goals from clause body
    #         new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
    #         query: Expr = self.query.apply_substitution(unifier)
    #         new_value = self.create_new_value(clause, new_facts)
    #         yield self.get_child(
    #             query=query,
    #             new_goals=new_goals,
    #             depth=self.depth + 1,
    #             value=new_value,
    #         )
    
    # def apply_clauses(self):
    # # If there are no goals left, the proof is complete
    #     if not self.goals:
    #         # Return a successful proof state with empty goals
    #         print("No more goals - proof complete!")
    #         yield self.get_child(
    #             new_goals=tuple(),
    #             depth=self.depth + 1,
    #             value=self.value
    #         )
    #         return

    #     first_goal, *remaining = self.goals
    #     print(f"Processing goal: {first_goal}")
        
    #     # Special handling for conjunctions
    #     if first_goal.functor == "," and first_goal.get_arity() > 0:
    #         print(f"Breaking down conjunction: {first_goal}")
    #         # Break down the conjunction into separate goals
    #         conjoined_goals = first_goal.arguments
    #         new_goals = conjoined_goals + tuple(remaining)
    #         print(f"New goals after breaking conjunction: {new_goals}")
    #         # Create a child proof with the broken-down goals
    #         yield self.get_child(
    #             new_goals=new_goals,
    #             depth=self.depth + 1,
    #             value=self.value
    #         )
    #         return
        
    #     # Original code for non-conjunctive goals
    #     matches = self.proof_tree.program.all_matches(first_goal)
    #     match_found = False
        
    #     for clause, unifier, new_facts in matches:
    #         match_found = True
    #         print(f"Found match with clause: {clause}")
            
    #         if clause.is_fact():
    #             # If we matched a fact, this goal is proven
    #             # Continue with remaining goals only
    #             print(f"Matched a fact, moving to remaining goals: {remaining}")
    #             new_goals = tuple(g.apply_substitution(unifier) for g in remaining)
    #             query = self.query.apply_substitution(unifier)
    #             new_value = self.create_new_value(clause, new_facts)
    #             yield self.get_child(
    #                 query=query,
    #                 new_goals=new_goals,
    #                 depth=self.depth + 1,
    #                 value=new_value,
    #             )
    #         else:
    #             # For rules, need to prove the body before continuing
    #             print(f"Matched a rule, adding body goals")
    #             # Apply unifier to the body goals first
    #             print(f"Clause body: {clause.arguments[1]}")
    #             print(f"{unifier=}")
    #             body = clause.arguments[1].apply_substitution(unifier)
    #             new_goals = body.arguments  # new goals from clause body with unifier applied
    #             new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
    #             query = self.query.apply_substitution(unifier)
    #             new_value = self.create_new_value(clause, new_facts)
    #             yield self.get_child(
    #                 query=query,
    #                 new_goals=new_goals,
    #                 depth=self.depth + 1,
    #                 value=new_value,
    #             )
        
    #     if not match_found:
    #         print(f"No matches found for: {first_goal}")
    
    # def apply_clauses(self):
    #     # If there are no goals left, the proof is complete
    #     if not self.goals:
    #         # Return a successful proof state with empty goals
    #         print("No more goals - proof complete!")
    #         yield self.get_child(
    #             new_goals=tuple(),
    #             depth=self.depth + 1,
    #             value=self.value
    #         )
    #         return

    #     first_goal, *remaining = self.goals
    #     print(f"Processing goal: {first_goal}")
        
    #     # Special handling for conjunctions
    #     if first_goal.functor == "," and first_goal.get_arity() > 0:
    #         print(f"Breaking down conjunction: {first_goal}")
    #         # Break down the conjunction into separate goals
    #         conjoined_goals = first_goal.arguments
    #         new_goals = conjoined_goals + tuple(remaining)
    #         print(f"New goals after breaking conjunction: {new_goals}")
    #         # Create a child proof with the broken-down goals
    #         yield self.get_child(
    #             new_goals=new_goals,
    #             depth=self.depth + 1,
    #             value=self.value
    #         )
    #         return
        
    #     # Original code for non-conjunctive goals
    #     matches = self.proof_tree.program.all_matches(first_goal)
    #     match_found = False
        
    #     for clause, unifier, new_facts in matches:
    #         match_found = True
    #         print(f"Found match with clause: {clause}")
            
    #         # Apply unifier to ALL remaining goals immediately
    #         # This ensures variable bindings from this goal propagate to all future goals
    #         updated_remaining = tuple(g.apply_substitution(unifier) for g in remaining)
            
    #         if clause.is_fact():
    #             # If we matched a fact, this goal is proven
    #             # Continue with remaining goals only with substitutions applied
    #             print(f"Matched a fact, moving to remaining goals: {updated_remaining}")
    #             query = self.query.apply_substitution(unifier)
    #             new_value = self.create_new_value(clause, new_facts)
    #             yield self.get_child(
    #                 query=query,
    #                 new_goals=updated_remaining,
    #                 depth=self.depth + 1,
    #                 value=new_value,
    #             )
    #         else:
    #             # For rules, need to prove the body before continuing
    #             print(f"Matched a rule, adding body goals")
    #             # Apply unifier to the body goals first
    #             print(f"Clause body: {clause.arguments[1]}")
    #             print(f"{unifier=}")
    #             body = clause.arguments[1].apply_substitution(unifier)
    #             # Apply the unifier to body goals and add remaining goals
    #             if body.functor == "," and body.get_arity() > 0:
    #                 # If body is a conjunction, extract its arguments
    #                 new_goals = body.arguments + updated_remaining
    #             else:
    #                 # Otherwise, add body as a single goal
    #                 new_goals = (body,) + updated_remaining
                    
    #             query = self.query.apply_substitution(unifier)
    #             new_value = self.create_new_value(clause, new_facts)
    #             yield self.get_child(
    #                 query=query,
    #                 new_goals=new_goals,
    #                 depth=self.depth + 1,
    #                 value=new_value,
    #             )
        
    #     if not match_found:
    #         print(f"No matches found for: {first_goal}")

    def get_child(
        self,
        query: Optional[Expr] = None,
        new_goals: tuple[Expr, ...] = tuple(),
        depth: Optional[int] = None,
        value: Optional[Value] = None,
        bindings: Optional[dict] = None,  # Add bindings parameter
    ):
        # If no new query is provided but we have bindings, apply them to the current query
        if query is None and bindings:
            query = self.query.apply_substitution(bindings)
        else:
            query = self.query if query is None else query
            
        # Combine existing bindings with new bindings
        combined_bindings = dict(self.current_bindings)  # Make a copy
        if bindings:
            combined_bindings.update(bindings)
            
        return ProofDebug(
            query=query,
            goals=new_goals,
            depth=self.depth if depth is None else depth,
            proof_tree=self.proof_tree,
            value=self.value if value is None else value,  # Move after proof_tree
            bindings=combined_bindings,
        )
    
    ##3.7 thinking version
    def apply_clauses(self):
        # If there are no goals left, the proof is complete
        if not self.goals:
            # Return a successful proof state with empty goals
            print("No more goals - proof complete!")
            yield self.get_child(
                new_goals=tuple(),
                depth=self.depth + 1,
                value=self.value,
                bindings=self.current_bindings  # Pass along current bindings
            )
            return

        first_goal, *remaining = self.goals
        # print(f"Processing goal: {first_goal}")
        
        # Special handling for conjunctions
        if first_goal.functor == "," and first_goal.get_arity() > 0:
            # print(f"Breaking down conjunction: {first_goal}")
            # Break down the conjunction into separate goals
            conjoined_goals = first_goal.arguments
            # Apply current bindings to all goals
            conjoined_goals = tuple(g.apply_substitution(self.current_bindings) for g in conjoined_goals)
            remaining_goals = tuple(g.apply_substitution(self.current_bindings) for g in remaining)
            new_goals = conjoined_goals + remaining_goals
            # print(f"New goals after breaking conjunction: {new_goals}")
            # Create a child proof with the broken-down goals
            yield self.get_child(
                new_goals=new_goals,
                depth=self.depth + 1,
                value=self.value,
                bindings=self.current_bindings
            )
            return
        
        # # NEW PART: Check if this goal can be satisfied by soft unifications
        # if self._can_satisfy_via_soft_unification(first_goal):
        #     print(f"Goal can be satisfied via soft unification: {first_goal}")
        #     # Create a new proof state with this goal removed and an adjusted probability
        #     new_value = self._adjust_value_for_soft_satisfaction(first_goal)
        #     yield self.get_child(
        #         new_goals=tuple(remaining),
        #         depth=self.depth + 1,
        #         value=new_value,
        #         bindings=self.current_bindings
        #     )

        
        
        # Original code for non-conjunctive goals
        matches = self.proof_tree.program.all_matches(first_goal)
        match_found = False
        
        for clause, unifier, new_facts in matches:
            match_found = True
            print(f"Found match with clause: {clause}")
            
            # Create updated bindings by combining current bindings with new unifier
            updated_bindings = dict(self.current_bindings)
            updated_bindings.update(unifier)
            
            if clause.is_fact():
                # If we matched a fact, this goal is proven
                # Continue with remaining goals only
                print(f"Matched a fact, moving to remaining goals: {remaining}")
                # Apply updated bindings to remaining goals
                new_goals = tuple(g.apply_substitution(updated_bindings) for g in remaining)
                query = self.query.apply_substitution(updated_bindings)
                new_value = self.create_new_value(clause, new_facts)
                yield self.get_child(
                    query=query,
                    new_goals=new_goals,
                    depth=self.depth + 1,
                    value=new_value,
                    bindings=updated_bindings
                )
            else:
                # For rules, need to prove the body before continuing
                print(f"Matched a rule, adding body goals")
                # Apply unifier to the body goals first
                print(f"Clause body: {clause.arguments[1]}")
                print(f"{unifier=}")
                # Apply the updated bindings to both body and remaining goals
                body = clause.arguments[1].apply_substitution(updated_bindings)
                new_goals = body.arguments + tuple(g.apply_substitution(updated_bindings) for g in remaining)
                query = self.query.apply_substitution(updated_bindings)
                new_value = self.create_new_value(clause, new_facts)
                yield self.get_child(
                    query=query,
                    new_goals=new_goals,
                    depth=self.depth + 1,
                    value=new_value,
                    bindings=updated_bindings
                )
        
        if not match_found:
            print(f"No matches found for: {first_goal}")
    
    def _debug_print(self, message, level=0):
        indent = "  " * (self.depth + level)
        print(f"{indent}DEBUG: {message}")