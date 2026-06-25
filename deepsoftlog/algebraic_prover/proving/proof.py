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
        

    def is_complete(self) -> bool:
        """Enhanced is_complete with diagnostic information"""
        # Check for empty goals
        is_goals_empty = len(self.goals) == 0
        
        # Diagnostic info
            
        #     # Check special case of object-only goals
        
        return is_goals_empty



    def nb_goals(self) -> int:
        
        return len(self.goals)

    def get_algebra(self) -> Algebra:
        return self.proof_tree.algebra

    def get_children(self) -> Iterable["Proof"]:
        if self.goals[0].functor == "\\+":
            yield from self.negation_node()
        else:
            yield from self.apply_clauses()

    def negation_node(self):
        negated_goal = self.goals[0].arguments[0]
        matches = self.proof_tree.program.all_matches(negated_goal)
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

    #     first_goal, *remaining = self.goals
        
            
    #         # For rules, handle the body correctly based on its structure
    #             # Extract body properly, handling the structure of conjunctive goals
                
    #             # Apply the unifier to each goal in the body
                
    #             # Add remaining goals with unifier applied
    #             new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
                
                
    #             yield self.get_child(
    #             )
    #             # Just continue with remaining goals
                
    #             yield self.get_child(
    #             )
    #     first_goal, *remaining = self.goals
    #         new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
    #         query: Expr = self.query.apply_substitution(unifier)
    #         yield self.get_child(
    #         )
    
    #     first_goal, *remaining = self.goals
        
    #     # Special handling for conjunctions
    #         # Break down the conjunction into separate goals
    #         # Create a child proof with the broken-down goals
    #         yield self.get_child(
    #         )
        
    #     # Original code for non-conjunctive goals
    #         new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
    #         query: Expr = self.query.apply_substitution(unifier)
    #         yield self.get_child(
    #         )
    
    def apply_clauses(self):
        """Enhanced apply_clauses with diagnostic tracing"""
        
        # Diagnostic for empty goals
        if not self.goals:
            yield self  # Simply return this proof as it's already complete
            return
        
        # Diagnostic for object-only goals
        if all(g.functor == "object" for g in self.goals):
            if hasattr(self.value, 'pos_facts'):
                
                # This is where we need to understand why these aren't being completed
                if hasattr(self, 'get_child'):
                    # Generate a child with empty goals for analysis
                    test_child = self.get_child(new_goals=tuple())
        
        # Standard processing
        first_goal, *remaining = self.goals
        
        # Special handling for conjunctions
        if first_goal.functor == "," and first_goal.get_arity() > 0:
            conjoined_goals = first_goal.arguments
            new_goals = conjoined_goals + tuple(remaining)
            new_child = self.get_child(new_goals=new_goals, depth=self.depth + 1, value=self.value)
            yield new_child
            return
        
        # Process matches
        matches = list(self.proof_tree.program.all_matches(first_goal))
        
        for i, (clause, unifier, new_facts) in enumerate(matches):
            
            if clause.is_fact():
                # Fact match - continue with remaining goals
                new_goals = tuple(g.apply_substitution(unifier) for g in remaining)
                query = self.query.apply_substitution(unifier)
                new_value = self.create_new_value(clause, new_facts)
                
                # Diagnostic for object-only remaining goals
                if new_goals and all(g.functor == "object" for g in new_goals):
                    if hasattr(new_value, 'pos_facts'):
                        pass
                
                child = self.get_child(query=query, new_goals=new_goals, depth=self.depth+1, value=new_value)
                yield child
            else:
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
                yield child


    
    def create_new_value(self, clause, new_facts):
        if new_facts:
            pass
        
        new_facts_value = self.get_algebra().reduce_mul_value_pos(new_facts)
        
        new_value = self.get_algebra().multiply(self.value, new_facts_value)
        
        if clause.is_annotated():
            clause_value = self.get_algebra().value_pos(clause)
            new_value = self.get_algebra().multiply_value_pos(new_value, clause)
        
        # # Check for numerical issues
        
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
    
    #     first_goal, *remaining = self.goals
    #         new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
    #         query: Expr = self.query.apply_substitution(unifier)
    #         yield self.get_child(
    #         )
    
    # # If there are no goals left, the proof is complete
    #         # Return a successful proof state with empty goals
    #         yield self.get_child(
    #         )

    #     first_goal, *remaining = self.goals
        
    #     # Special handling for conjunctions
    #         # Break down the conjunction into separate goals
    #         # Create a child proof with the broken-down goals
    #         yield self.get_child(
    #         )
        
    #     # Original code for non-conjunctive goals
        
            
    #             # If we matched a fact, this goal is proven
    #             # Continue with remaining goals only
    #             yield self.get_child(
    #             )
    #             # For rules, need to prove the body before continuing
    #             # Apply unifier to the body goals first
    #             new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
    #             yield self.get_child(
    #             )
        
    
    #     # If there are no goals left, the proof is complete
    #         # Return a successful proof state with empty goals
    #         yield self.get_child(
    #         )

    #     first_goal, *remaining = self.goals
        
    #     # Special handling for conjunctions
    #         # Break down the conjunction into separate goals
    #         # Create a child proof with the broken-down goals
    #         yield self.get_child(
    #         )
        
    #     # Original code for non-conjunctive goals
        
            
    #         # Apply unifier to ALL remaining goals immediately
    #         # This ensures variable bindings from this goal propagate to all future goals
            
    #             # If we matched a fact, this goal is proven
    #             # Continue with remaining goals only with substitutions applied
    #             yield self.get_child(
    #             )
    #             # For rules, need to prove the body before continuing
    #             # Apply unifier to the body goals first
    #             # Apply the unifier to body goals and add remaining goals
    #                 # If body is a conjunction, extract its arguments
    #                 # Otherwise, add body as a single goal
                    
    #             yield self.get_child(
    #             )
        

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
            yield self.get_child(
                new_goals=tuple(),
                depth=self.depth + 1,
                value=self.value,
                bindings=self.current_bindings  # Pass along current bindings
            )
            return

        first_goal, *remaining = self.goals
        
        # Special handling for conjunctions
        if first_goal.functor == "," and first_goal.get_arity() > 0:
            # Break down the conjunction into separate goals
            conjoined_goals = first_goal.arguments
            # Apply current bindings to all goals
            conjoined_goals = tuple(g.apply_substitution(self.current_bindings) for g in conjoined_goals)
            remaining_goals = tuple(g.apply_substitution(self.current_bindings) for g in remaining)
            new_goals = conjoined_goals + remaining_goals
            # Create a child proof with the broken-down goals
            yield self.get_child(
                new_goals=new_goals,
                depth=self.depth + 1,
                value=self.value,
                bindings=self.current_bindings
            )
            return
        
        # # NEW PART: Check if this goal can be satisfied by soft unifications
        #     # Create a new proof state with this goal removed and an adjusted probability
        #     yield self.get_child(
        #     )

        
        
        # Original code for non-conjunctive goals
        matches = self.proof_tree.program.all_matches(first_goal)
        match_found = False
        
        for clause, unifier, new_facts in matches:
            match_found = True
            
            # Create updated bindings by combining current bindings with new unifier
            updated_bindings = dict(self.current_bindings)
            updated_bindings.update(unifier)
            
            if clause.is_fact():
                # If we matched a fact, this goal is proven
                # Continue with remaining goals only
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
                # Apply unifier to the body goals first
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
        
    
    def _debug_print(self, message, level=0):
        indent = "  " * (self.depth + level)
        


