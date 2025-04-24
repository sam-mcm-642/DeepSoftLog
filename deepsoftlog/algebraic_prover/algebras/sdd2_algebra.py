from typing import Union

from deepsoftlog.algebraic_prover.algebras.sdd_algebra import SddAlgebra, SddFormula
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import CompoundAlgebra, Algebra, Value
from deepsoftlog.algebraic_prover.terms.expression import Fact, Expr
import torch


class ConjoinedFacts:
    def __init__(
        self, pos_facts: set[Expr], neg_facts: set[Expr], sdd_algebra: SddAlgebra
    ):
        self.pos_facts = pos_facts
        self.neg_facts = neg_facts
        self._sdd_algebra = sdd_algebra

    def __and__(self, other: "ConjoinedFacts") -> Union["ConjoinedFacts", SddFormula]:
        new_pos_facts = self.pos_facts | other.pos_facts
        new_neg_facts = self.neg_facts | other.neg_facts
        if len(new_pos_facts & new_neg_facts) != 0:
            return self._sdd_algebra.zero()
        return ConjoinedFacts(new_pos_facts, new_neg_facts, self._sdd_algebra)

    # def evaluate(self, algebra: Algebra):
    #     return algebra.reduce_mul_value(self.pos_facts, self.neg_facts)
    
    # def evaluate(self, algebra: Algebra):
    #     print(f"ConjoinedFacts.evaluate called with pos_facts={self.pos_facts}")
        
    #     # If there are no positive facts, this might be causing the -inf
    #     if not self.pos_facts:
    #         print("Warning: No positive facts to evaluate")
    #         # Return a very small probability instead of zero/inf
    #         return torch.tensor(-20.0)
        
    #     result = algebra.reduce_mul_value(self.pos_facts, self.neg_facts)
    #     print(f"reduce_mul_value result: {result}")
    
        # return result
        
    # def evaluate(self, algebra: Algebra):
    #     print(f"ConjoinedFacts.evaluate called with pos_facts={self.pos_facts}")        
    #     result = algebra.reduce_mul_value(self.pos_facts, self.neg_facts)
    #     print(f"reduce_mul_value result: {result}")
    #     return result
    
    def evaluate(self, algebra: Algebra):
        print(f"ConjoinedFacts.evaluate called with pos_facts={self.pos_facts}")
        
        # Check for empty facts
        if not self.pos_facts:
            print("WARNING: Empty positive facts collection")
            
        # Try to detect problematic fact patterns
        small_probs = [f for f in self.pos_facts if hasattr(f, 'get_probability') and f.get_probability() < 0.001]
        if small_probs:
            print(f"WARNING: Very small probabilities detected: {small_probs}")
        
        result = algebra.reduce_mul_value(self.pos_facts, self.neg_facts)
        print(f"reduce_mul_value result: {result}")
        
        # Check result for numerical issues
        if isinstance(result, torch.Tensor):
            if torch.isneginf(result):
                print("CRITICAL: Evaluation resulted in -inf")
            elif torch.isnan(result):
                print("CRITICAL: Evaluation resulted in NaN")
                
        return result

    def __str__(self):
        return f"ConjoinedFacts({self.pos_facts}, {self.neg_facts})"

    def __repr__(self):
        return f"ConjoinedFacts({self.pos_facts}, {self.neg_facts})"


class DnfAlgebra(CompoundAlgebra[Union[ConjoinedFacts, SddFormula]]):
    """
    Like the Sdd Algebra, but uses sets for simple conjunctions.
    This can be considerably faster, especially for programs without
    negation on rules. (in which case the knowledge compilation
    is only performed after all proofs are found).
    """

    def __init__(self, eval_algebra: Algebra):
        super().__init__(eval_algebra)
        self._sdd_algebra = SddAlgebra(eval_algebra)

    def value_pos(self, fact: Fact) -> Value:
        return self._as_conjoined_facts(pos_facts={fact})

    def value_neg(self, fact: Fact) -> Value:
        return self._as_conjoined_facts(neg_facts={fact})

    def multiply(self, v1: Value, v2: Value) -> Value:
        if not isinstance(v1, ConjoinedFacts) or not isinstance(v2, ConjoinedFacts):
            v1 = self._as_sdd(v1)
            v2 = self._as_sdd(v2)
        return v1 & v2

    def reduce_mul_value_pos(self, facts) -> Value:
        facts = {f for f in facts if f.is_annotated()}
        return self._as_conjoined_facts(facts)

    def add(self, value1: Value, value2: Value) -> Value:
        return self._as_sdd(value1) | self._as_sdd(value2)

    def one(self) -> Value:
        return self._as_conjoined_facts()

    def zero(self) -> Value:
        return self._sdd_algebra.zero()

    def reset(self):
        self._sdd_algebra.reset()

    # def _as_sdd(self, value):
    #     if isinstance(value, ConjoinedFacts):
    #         return value.evaluate(self._sdd_algebra)
    #     return value
    
    # def _as_sdd(self, value):
    #     if isinstance(value, ConjoinedFacts):
    #         print(f"Converting ConjoinedFacts to SDD: pos_facts={value.pos_facts}, neg_facts={value.neg_facts}")
    #         result = value.evaluate(self._sdd_algebra)
    #         print(f"SDD evaluation result: {result}")
            
    #         # Add a floor to prevent -inf values
    #         if isinstance(result, torch.Tensor) and torch.isneginf(result):
    #             print(f"Found -inf result, replacing with floor value")
    #             result = torch.tensor(-20.0)  # A small log probability, not -inf
            
    #         return result
    #     return value
    
    def _as_sdd(self, value):
        if isinstance(value, ConjoinedFacts):
            print(f"Converting ConjoinedFacts to SDD: pos_facts={value.pos_facts}, neg_facts={value.neg_facts}")
            
            # Pre-check for potential issues
            if not value.pos_facts:
                print("WARNING: Converting empty ConjoinedFacts to SDD")
                
            result = value.evaluate(self._sdd_algebra)
            print(f"SDD evaluation result: {result}")
            
            # Add numerical safety checks
            if isinstance(result, torch.Tensor):
                if torch.isneginf(result):
                    print(f"FIXING: -inf result, replacing with floor value -20.0")
                    result = torch.tensor(-20.0)
                elif torch.isnan(result):
                    print(f"FIXING: NaN result, replacing with floor value -20.0")
                    result = torch.tensor(-20.0)
            
            return result
        return value

    
    # def evaluate(self, value):
    #     print(f"DnfAlgebra.evaluate called with value: {value}, type: {type(value)}")
    #     if isinstance(value, ConjoinedFacts):
    #         sdd_value = self._as_sdd(value)
    #         print(f"After _as_sdd: {sdd_value}")
    #         result = self._eval_algebra.evaluate(sdd_value)
    #         print(f"After _eval_algebra.evaluate: {result}")
    #         return result
    #     elif hasattr(value, 'evaluate'):
    #         print(f"Value has evaluate method")
    #         return value.evaluate(self._eval_algebra)
    #     else:
    #         print(f"Direct evaluation by _eval_algebra")
    #         return self._eval_algebra.evaluate(value)

    def _as_conjoined_facts(self, pos_facts=None, neg_facts=None):
        pos_facts = pos_facts or set()
        neg_facts = neg_facts or set()
        return ConjoinedFacts(pos_facts, neg_facts, self._sdd_algebra)
