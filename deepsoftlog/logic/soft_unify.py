from typing import Optional

from deepsoftlog.algebraic_prover.proving.unify import replace_all_occurrences
from deepsoftlog.algebraic_prover.terms.variable import Variable
from deepsoftlog.algebraic_prover.terms.probability_annotation import LogProbabilisticExpr
from deepsoftlog.algebraic_prover.terms.expression import Expr


def get_unify_fact(term1: Expr, term2: Expr, store, metric: str) -> Expr:
    if term1 < term2:
        term1, term2 = term2, term1
    prob = store.soft_unify_score(term1, term2, metric)
    fact = Expr("k", term1, term2)
    return LogProbabilisticExpr(prob, ":-", fact, Expr(","), infix=True)



def is_soft(e: Expr):
    result = e.get_predicate() == ("~", 1)
    return result


def look_for_rr(x) -> int:
    if isinstance(x, Variable):
        return 0
    elif isinstance(x, list) or isinstance(x, tuple):
        return sum(look_for_rr(t) for t in x)
    else:  # if isinstance(x, Expr):
        return x.functor.startswith("rr") + look_for_rr(x.arguments)

def look_for_oobj(x) -> int:
    if isinstance(x, Variable):
        return 0
    elif isinstance(x, list) or isinstance(x, tuple):
        return sum(look_for_oobj(t) for t in x)
    else:  # if isinstance(x, Expr):
        return x.functor.startswith("oobj") + look_for_oobj(x.arguments)

#     # No occurs check
#             s, t = substitution[i]
#                 substitution[i] = (t, s)

#                     del substitution[i]

#                     s, t = s.arguments[0], t.arguments[0]
#                         substitution[i] = (s, t)
#                         substitution[i] = (t, s)
#                             soft_unifies.add(get_unify_fact(s, t, store, metric))
#                         del substitution[i]

#                     # can't hard unify



    
#     # For each argument, print its type and structure
    
    
#     # No occurs check
#             s, t = substitution[i]
            
#                 substitution[i] = (t, s)

#                     del substitution[i]

                
#                     s_inner, t_inner = s.arguments[0], t.arguments[0]
                    
#                         substitution[i] = (s_inner, t_inner)
#                         substitution[i] = (t_inner, s_inner)
#                             soft_unifies.add(get_unify_fact(s_inner, t_inner, store, metric))
#                         del substitution[i]

#                     # can't hard unify
                



##3.7 thinking version

#     """
#     Most General Unifier with support for soft terms (~).
#     Returns a tuple of (substitution dict, set of soft unification facts) or None if unification fails.
#     """
    
#     # Debug output for arguments
    
    
#     # Early exit conditions
    
#     # Initialize substitution list and soft facts set
    
#     # Main unification loop
#             s, t = substitution[i]
            
#             # Case 1: Swap variable/non-variable pairs to ensure variable is on left
#                 substitution[i] = (t, s)

#             # Case 2: Handle variable on left side
#                 # Case 2.1: Identity - remove redundant mapping
#                     del substitution[i]
                
#                 # Case 2.2: Check for occurs (variable appearing in the term it's bound to)
                
#                 # Case 2.3: Apply substitution throughout
            
#             # Case 3: Handle expressions on both sides
                
#                 # Case 3.1: Both are soft terms (~)
#                     s_inner, t_inner = s.arguments[0], t.arguments[0]
                    
#                     # Case 3.1.1: First inner term is variable
#                         substitution[i] = (s_inner, t_inner)
#                     # Case 3.1.2: Second inner term is variable
#                         substitution[i] = (t_inner, s_inner)
#                     # Case 3.1.3: Both inner terms are ground
#                             soft_unifies.add(get_unify_fact(s_inner, t_inner, store, metric))
#                         del substitution[i]
#                     # Case 3.1.4: Error case - can't soft unify non-ground terms
                
#                 # Case 3.2: Different predicates - can't unify
                
#                 # Case 3.3: Same predicate - unify arguments
                    
            
#             # Move to next pair if no changes
#             i += 1

#     # Convert substitution list to proper dictionary
#     # Ensure only variables appear as keys
#             result_dict[k] = v
    


def soft_mgu(term1: Expr, term2: Expr, store, metric, soft_cache=None) -> Optional[tuple[dict, set]]:
    """Most General Unifier with support for soft terms (~)."""
    
    # Early exit conditions
    if look_for_rr([term1, term2]) > 1 or look_for_oobj([term1, term2]) > 2:
        print("Early exit: Too many rr or oobj terms")
        return None
    
    # Initialize substitution list and soft facts set
    substitution = [(term1, term2)]
    soft_unifies = set()
    changes = True
    
    # Main unification loop
    while changes:
        changes = False
        i = 0
        while i < len(substitution):
            s, t = substitution[i]
            
            # Case 1: Swap variable/non-variable pairs
            if type(t) is Variable and type(s) is not Variable:
                substitution[i] = (t, s)
                changes = True
                break

            # Case 2: Handle variable on left side
            if type(s) is Variable:
                if t == s:
                    del substitution[i]
                    changes = True
                    break
                
                if isinstance(t, Expr) and s in t.all_variables():
                    return None
                
                new_substitution = replace_all_occurrences(s, t, i, substitution)
                if new_substitution is not None:
                    substitution = new_substitution
                    changes = True
                    break
            
            # Case 3: Handle expressions on both sides
            if isinstance(s, Expr) and isinstance(t, Expr):
                # Case 3.1: Both are soft terms (~)
                if is_soft(s) and is_soft(t):
                    s_inner, t_inner = s.arguments[0], t.arguments[0]
                    
                    if isinstance(s_inner, Variable):
                        substitution[i] = (s_inner, t_inner)
                        changes = True
                        break
                    elif isinstance(t_inner, Variable):
                        substitution[i] = (t_inner, s_inner)
                        changes = True
                        break
                    elif s_inner.is_ground() and t_inner.is_ground():
                        if s_inner != t_inner:
                            # Create a canonical key for caching
                            terms = sorted([str(s_inner), str(t_inner)])
                            pair_key = (terms[0], terms[1])
                            
                            # Check cache for existing unification
                            if soft_cache is not None and pair_key in soft_cache and 1 == 0:
                                print(f"Reusing cached soft unification for {s_inner} and {t_inner}")
                                soft_unifies.add(soft_cache[pair_key])
                            else:
                                fact = get_unify_fact(s_inner, t_inner, store, metric)
                                soft_unifies.add(fact)
                                
                                # Cache this soft unification
                                if soft_cache is not None:
                                    soft_cache[pair_key] = fact
                        
                        del substitution[i]
                        changes = True
                        if len(soft_unifies) > 0:
                            # Check probability values of soft facts
                            for fact in soft_unifies:
                                if hasattr(fact, 'get_probability'):
                                    prob = fact.get_probability()
                                    if prob < 0.001:
                                        pass
                        break
                    else:
                        raise Exception(f"Soft unification of non-ground terms `{s_inner}` and `{t_inner}` is illegal")
                
                if s.get_predicate() != t.get_predicate():
                    return None
                
                new_substitution = list(zip(s.arguments, t.arguments))
                substitution = substitution[:i] + new_substitution + substitution[i+1:]
                changes = True
                break
            
            i += 1

    # Convert substitution list to proper dictionary
    result_dict = {}
    for k, v in substitution:
        if isinstance(k, Variable):
            result_dict[k] = v
    
    return result_dict, soft_unifies

###NEWEST VERSION
#     """
#     Enhanced Most General Unifier function that handles soft unification.
#     Tracks both hard unification substitutions and soft unifications.
#     """

#     # Early check for relations that we want to avoid
        
#     # Initialize the substitution list and soft unifications set
    
#     # Process the substitution list until no more changes are made
#         iteration += 1
        
                
#             s, t = substitution[i]
            
#             # If t is a variable and s is not, swap them to ensure variables are on the left
#                 substitution[i] = (t, s)

#             # Handle variable on left side
#                     # Remove identity substitution
#                     del substitution[i]
                
#                 # Check if the variable occurs in the term (occurs check)
                
#                 # Replace all occurrences of the variable in other substitutions

#             # Both sides are expressions
#                 # Special handling for soft terms (terms with ~ functor)
#                     s_inner, t_inner = s.arguments[0], t.arguments[0]
                    
#                         substitution[i] = (s_inner, t_inner)
#                         substitution[i] = (t_inner, s_inner)
#                             soft_unifies.add(get_unify_fact(s_inner, t_inner, store, metric))
#                         del substitution[i]

#                 # Regular hard unification for non-soft terms
                
#                     # For constants with no arguments, just remove this pair
#                     del substitution[i]

#     # Convert the list of substitutions to a dictionary
#             result_dict[var] = term
    