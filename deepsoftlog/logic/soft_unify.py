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


# def is_soft(e: Expr):
#     return e.get_predicate() == ("~", 1)

def is_soft(e: Expr):
    result = e.get_predicate() == ("~", 1)
    # print(f"Checking if {e} is soft: {result}, predicate: {e.get_predicate()}")
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

# def soft_mgu(term1: Expr, term2: Expr, store, metric) -> Optional[tuple[dict, set]]:
#     if look_for_rr([term1, term2]) > 1 or look_for_oobj([term1, term2]) > 1:
#         return
#     # No occurs check
#     substitution = [(term1, term2)]
#     soft_unifies = set()
#     changes = True
#     while changes:
#         changes = False
#         for i in range(len(substitution)):
#             s, t = substitution[i]
#             if type(t) is Variable and type(s) is not Variable:
#                 substitution[i] = (t, s)
#                 changes = True
#                 break

#             if type(s) is Variable:
#                 if t == s:
#                     del substitution[i]
#                     changes = True
#                     break
#                 new_substitution = replace_all_occurrences(s, t, i, substitution)
#                 if new_substitution is not None:
#                     substitution = new_substitution
#                     changes = True
#                     break

#             if isinstance(s, Expr) and isinstance(t, Expr):
#                 if is_soft(s) and is_soft(t):
#                     s, t = s.arguments[0], t.arguments[0]
#                     if isinstance(s, Variable):
#                         substitution[i] = (s, t)
#                     elif isinstance(t, Variable):
#                         substitution[i] = (t, s)
#                     elif s.is_ground() and t.is_ground():
#                         if s != t:
#                             soft_unifies.add(get_unify_fact(s, t, store, metric))
#                         del substitution[i]
#                     else:
#                         raise Exception(f"Soft unification of non-ground terms `{s}` and `{t}` is illegal")
#                     changes = True
#                     break

#                 if s.get_predicate() != t.get_predicate():
#                     # can't hard unify
#                     return None
#                 new_substitution = list(zip(s.arguments, t.arguments))
#                 substitution = substitution[:i] + new_substitution + substitution[i+1:]
#                 changes = True
#                 break

#     return dict(substitution), soft_unifies


# def soft_mgu(term1: Expr, term2: Expr, store, metric) -> Optional[tuple[dict, set]]:
#     print(f"Attempting soft_mgu between {term1} and {term2}")
#     print(f"Term1 type: {type(term1)}, Term2 type: {type(term2)}")
    
#     # For each argument, print its type and structure
#     if isinstance(term1, Expr) and hasattr(term1, 'arguments'):
#         for i, arg in enumerate(term1.arguments):
#             print(f"Term1 arg {i}: {arg}, Type: {type(arg)}")
#             if isinstance(arg, Expr) and hasattr(arg, 'arguments'):
#                 print(f"  - Functor: {arg.functor}")
#                 print(f"  - Arguments: {arg.arguments}")
    
#     if isinstance(term2, Expr) and hasattr(term2, 'arguments'):
#         for i, arg in enumerate(term2.arguments):
#             print(f"Term2 arg {i}: {arg}, Type: {type(arg)}")
#             if isinstance(arg, Expr) and hasattr(arg, 'arguments'):
#                 print(f"  - Functor: {arg.functor}")
#                 print(f"  - Arguments: {arg.arguments}")
    
#     if look_for_rr([term1, term2]) > 1 or look_for_oobj([term1, term2]) > 2:
#         return
#     # No occurs check
#     substitution = [(term1, term2)]
#     soft_unifies = set()
#     changes = True
#     while changes:
#         changes = False
#         for i in range(len(substitution)):
#             s, t = substitution[i]
#             print(f"Checking substitution pair: {s}, {t}")
            
#             if type(t) is Variable and type(s) is not Variable:
#                 print(f"Swapping variable {t} with non-variable {s}")
#                 substitution[i] = (t, s)
#                 changes = True
#                 break

#             if type(s) is Variable:
#                 if t == s:
#                     print(f"Removing identical var-var pair: {s}={t}")
#                     del substitution[i]
#                     changes = True
#                     break
#                 print(f"Replacing occurrences of {s} with {t}")
#                 new_substitution = replace_all_occurrences(s, t, i, substitution)
#                 if new_substitution is not None:
#                     substitution = new_substitution
#                     changes = True
#                     break

#             if isinstance(s, Expr) and isinstance(t, Expr):
#                 print(f"Both are expressions: {s.get_predicate()} and {t.get_predicate()}")
#                 print(f"Checking if soft: {is_soft(s)} and {is_soft(t)}")
                
#                 if is_soft(s) and is_soft(t):
#                     print(f"Both are soft terms: {s} and {t}")
#                     s_inner, t_inner = s.arguments[0], t.arguments[0]
#                     print(f"Inner terms: {s_inner} and {t_inner}")
                    
#                     if isinstance(s_inner, Variable):
#                         print(f"First inner term is variable: {s_inner}")
#                         substitution[i] = (s_inner, t_inner)
#                     elif isinstance(t_inner, Variable):
#                         print(f"Second inner term is variable: {t_inner}")
#                         substitution[i] = (t_inner, s_inner)
#                     elif s_inner.is_ground() and t_inner.is_ground():
#                         print(f"Both inner terms are ground")
#                         if s_inner != t_inner:
#                             print(f"Creating soft unification fact for {s_inner} and {t_inner}")
#                             soft_unifies.add(get_unify_fact(s_inner, t_inner, store, metric))
#                         del substitution[i]
#                     else:
#                         raise Exception(f"Soft unification of non-ground terms `{s_inner}` and `{t_inner}` is illegal")
#                     changes = True
#                     break

#                 if s.get_predicate() != t.get_predicate():
#                     print(f"Predicates don't match: {s.get_predicate()} vs {t.get_predicate()}")
#                     # can't hard unify
#                     return None
                
#                 print(f"Creating substitution pairs for arguments")
#                 new_substitution = list(zip(s.arguments, t.arguments))
#                 substitution = substitution[:i] + new_substitution + substitution[i+1:]
#                 changes = True
#                 break

#     print(f"Final substitution: {dict(substitution)}")
#     print(f"Soft unifies: {soft_unifies}")
#     return dict(substitution), soft_unifies


##3.7 thinking version

# def soft_mgu(term1: Expr, term2: Expr, store, metric) -> Optional[tuple[dict, set]]:
#     """
#     Most General Unifier with support for soft terms (~).
#     Returns a tuple of (substitution dict, set of soft unification facts) or None if unification fails.
#     """
#     print(f"Attempting soft_mgu between {term1} and {term2}")
#     print(f"Term1 type: {type(term1)}, Term2 type: {type(term2)}")
    
#     # Debug output for arguments
#     if isinstance(term1, Expr) and hasattr(term1, 'arguments'):
#         for i, arg in enumerate(term1.arguments):
#             print(f"Term1 arg {i}: {arg}, Type: {type(arg)}")
#             if isinstance(arg, Expr) and hasattr(arg, 'arguments'):
#                 print(f"  - Functor: {arg.functor}")
#                 print(f"  - Arguments: {arg.arguments}")
    
#     if isinstance(term2, Expr) and hasattr(term2, 'arguments'):
#         for i, arg in enumerate(term2.arguments):
#             print(f"Term2 arg {i}: {arg}, Type: {type(arg)}")
#             if isinstance(arg, Expr) and hasattr(arg, 'arguments'):
#                 print(f"  - Functor: {arg.functor}")
#                 print(f"  - Arguments: {arg.arguments}")
    
#     # Early exit conditions
#     if look_for_rr([term1, term2]) > 1 or look_for_oobj([term1, term2]) > 2:
#         print("Early exit: Too many rr or oobj terms")
#         return None
    
#     # Initialize substitution list and soft facts set
#     substitution = [(term1, term2)]
#     soft_unifies = set()
#     changes = True
    
#     # Main unification loop
#     while changes:
#         changes = False
#         i = 0
#         while i < len(substitution):
#             s, t = substitution[i]
#             print(f"Checking substitution pair: {s}, {t}")
            
#             # Case 1: Swap variable/non-variable pairs to ensure variable is on left
#             if type(t) is Variable and type(s) is not Variable:
#                 print(f"Swapping variable {t} with non-variable {s}")
#                 substitution[i] = (t, s)
#                 changes = True
#                 break

#             # Case 2: Handle variable on left side
#             if type(s) is Variable:
#                 # Case 2.1: Identity - remove redundant mapping
#                 if t == s:
#                     print(f"Removing identical var-var pair: {s}={t}")
#                     del substitution[i]
#                     changes = True
#                     break
                
#                 # Case 2.2: Check for occurs (variable appearing in the term it's bound to)
#                 if isinstance(t, Expr) and s in t.all_variables():
#                     print(f"Occurs check failed: {s} appears in {t}")
#                     return None
                
#                 # Case 2.3: Apply substitution throughout
#                 print(f"Replacing occurrences of {s} with {t}")
#                 new_substitution = replace_all_occurrences(s, t, i, substitution)
#                 if new_substitution is not None:
#                     substitution = new_substitution
#                     changes = True
#                     break
            
#             # Case 3: Handle expressions on both sides
#             if isinstance(s, Expr) and isinstance(t, Expr):
#                 print(f"Both are expressions: {s.get_predicate()} and {t.get_predicate()}")
#                 print(f"Checking if soft: {is_soft(s)} and {is_soft(t)}")
                
#                 # Case 3.1: Both are soft terms (~)
#                 if is_soft(s) and is_soft(t):
#                     print(f"Both are soft terms: {s} and {t}")
#                     s_inner, t_inner = s.arguments[0], t.arguments[0]
#                     print(f"Inner terms: {s_inner} and {t_inner}")
                    
#                     # Case 3.1.1: First inner term is variable
#                     if isinstance(s_inner, Variable):
#                         print(f"First inner term is variable: {s_inner}")
#                         substitution[i] = (s_inner, t_inner)
#                         changes = True
#                         break
#                     # Case 3.1.2: Second inner term is variable
#                     elif isinstance(t_inner, Variable):
#                         print(f"Second inner term is variable: {t_inner}")
#                         substitution[i] = (t_inner, s_inner)
#                         changes = True
#                         break
#                     # Case 3.1.3: Both inner terms are ground
#                     elif s_inner.is_ground() and t_inner.is_ground():
#                         print(f"Both inner terms are ground")
#                         if s_inner != t_inner:
#                             print(f"Creating soft unification fact for {s_inner} and {t_inner}")
#                             soft_unifies.add(get_unify_fact(s_inner, t_inner, store, metric))
#                         del substitution[i]
#                         changes = True
#                         break
#                     # Case 3.1.4: Error case - can't soft unify non-ground terms
#                     else:
#                         raise Exception(f"Soft unification of non-ground terms `{s_inner}` and `{t_inner}` is illegal")
                
#                 # Case 3.2: Different predicates - can't unify
#                 if s.get_predicate() != t.get_predicate():
#                     print(f"Predicates don't match: {s.get_predicate()} vs {t.get_predicate()}")
#                     return None
                
#                 # Case 3.3: Same predicate - unify arguments
#                 print(f"Creating substitution pairs for arguments")
#                 if len(s.arguments) != len(t.arguments):
#                     print(f"Arity mismatch: {len(s.arguments)} vs {len(t.arguments)}")
#                     return None
                    
#                 new_substitution = list(zip(s.arguments, t.arguments))
#                 substitution = substitution[:i] + new_substitution + substitution[i+1:]
#                 changes = True
#                 break
            
#             # Move to next pair if no changes
#             i += 1

#     # Convert substitution list to proper dictionary
#     # Ensure only variables appear as keys
#     result_dict = {}
#     for k, v in substitution:
#         if isinstance(k, Variable):
#             result_dict[k] = v
    
#     print(f"Final substitution: {result_dict}")
#     print(f"Soft unifies: {soft_unifies}")
#     return result_dict, soft_unifies


def soft_mgu(term1: Expr, term2: Expr, store, metric, soft_cache=None) -> Optional[tuple[dict, set]]:
    """Most General Unifier with support for soft terms (~)."""
    # print(f"Attempting soft_mgu between {term1} and {term2}")
    
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
                            if soft_cache is not None and pair_key in soft_cache:
                                # print(f"Reusing cached soft unification for {s_inner} and {t_inner}")
                                soft_unifies.add(soft_cache[pair_key])
                            else:
                                # print(f"Creating new soft unification fact for {s_inner} and {t_inner}")
                                fact = get_unify_fact(s_inner, t_inner, store, metric)
                                soft_unifies.add(fact)
                                
                                # Cache this soft unification
                                if soft_cache is not None:
                                    soft_cache[pair_key] = fact
                        
                        del substitution[i]
                        changes = True
                        if len(soft_unifies) > 0:
                            print(f"SOFT UNIFICATION FACTS: {soft_unifies}")
                            # Check probability values of soft facts
                            for fact in soft_unifies:
                                if hasattr(fact, 'get_probability'):
                                    prob = fact.get_probability()
                                    if prob < 0.001:
                                        print(f"WARNING: Very small soft unification probability: {prob}")
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
    
    print(f"Final substitution: {result_dict}")
    print(f"Soft unifies: {soft_unifies}")
    return result_dict, soft_unifies

###NEWEST VERSION
# def soft_mgu(term1: Expr, term2: Expr, store, metric) -> Optional[tuple[dict, set]]:
#     """
#     Enhanced Most General Unifier function that handles soft unification.
#     Tracks both hard unification substitutions and soft unifications.
#     """
#     print(f"Attempting soft_mgu between {term1} and {term2}")

#     # Early check for relations that we want to avoid
#     if look_for_rr([term1, term2]) > 1 or look_for_oobj([term1, term2]) > 1:
#         print(f"Aborting unification due to multiple 'rr' or 'oobj' terms")
#         return None
        
#     # Initialize the substitution list and soft unifications set
#     substitution = [(term1, term2)]
#     soft_unifies = set()
#     changes = True
#     iteration = 0
    
#     # Process the substitution list until no more changes are made
#     while changes:
#         changes = False
#         iteration += 1
#         print(f"Unification iteration {iteration}, current subs: {substitution}")
        
#         for i in range(len(substitution)):
#             if i >= len(substitution):  # Safety check in case items were deleted
#                 break
                
#             s, t = substitution[i]
#             print(f"Examining pair: {s} and {t}")
            
#             # If t is a variable and s is not, swap them to ensure variables are on the left
#             if type(t) is Variable and type(s) is not Variable:
#                 print(f"Swapping variable {t} with non-variable {s}")
#                 substitution[i] = (t, s)
#                 changes = True
#                 break

#             # Handle variable on left side
#             if type(s) is Variable:
#                 if t == s:
#                     # Remove identity substitution
#                     print(f"Removing identical var-var pair: {s}={t}")
#                     del substitution[i]
#                     changes = True
#                     break
                
#                 # Check if the variable occurs in the term (occurs check)
#                 if isinstance(t, Expr) and s in t.all_variables():
#                     print(f"Occurs check failed: {s} occurs in {t}")
#                     return None
                
#                 # Replace all occurrences of the variable in other substitutions
#                 print(f"Replacing occurrences of {s} with {t}")
#                 new_substitution = replace_all_occurrences(s, t, i, substitution)
#                 if new_substitution is not None:
#                     substitution = new_substitution
#                     changes = True
#                     break

#             # Both sides are expressions
#             if isinstance(s, Expr) and isinstance(t, Expr):
#                 # Special handling for soft terms (terms with ~ functor)
#                 if is_soft(s) and is_soft(t):
#                     print(f"Both are soft terms: {s} and {t}")
#                     s_inner, t_inner = s.arguments[0], t.arguments[0]
#                     print(f"Inner terms: {s_inner} and {t_inner}")
                    
#                     if isinstance(s_inner, Variable):
#                         print(f"First inner term is variable: {s_inner}")
#                         substitution[i] = (s_inner, t_inner)
#                     elif isinstance(t_inner, Variable):
#                         print(f"Second inner term is variable: {t_inner}")
#                         substitution[i] = (t_inner, s_inner)
#                     elif s_inner.is_ground() and t_inner.is_ground():
#                         print(f"Both inner terms are ground")
#                         if s_inner != t_inner:
#                             print(f"Creating soft unification fact for {s_inner} and {t_inner}")
#                             soft_unifies.add(get_unify_fact(s_inner, t_inner, store, metric))
#                         del substitution[i]
#                     else:
#                         print(f"Soft unification of non-ground terms `{s_inner}` and `{t_inner}` is illegal")
#                         return None
#                     changes = True
#                     break

#                 # Regular hard unification for non-soft terms
#                 if s.get_predicate() != t.get_predicate():
#                     print(f"Predicates don't match: {s.get_predicate()} vs {t.get_predicate()}")
#                     return None
                
#                 print(f"Creating substitution pairs for arguments")
#                 new_substitution = list(zip(s.arguments, t.arguments))
#                 if len(new_substitution) > 0:  # Only make changes if there are arguments to process
#                     substitution = substitution[:i] + new_substitution + substitution[i+1:]
#                     changes = True
#                     break
#                 else:
#                     # For constants with no arguments, just remove this pair
#                     del substitution[i]
#                     changes = True
#                     break

#     # Convert the list of substitutions to a dictionary
#     result_dict = {}
#     for var, term in substitution:
#         if isinstance(var, Variable):
#             result_dict[var] = term
    
#     print(f"Final substitution: {result_dict}")
#     print(f"Soft unifies: {soft_unifies}")
#     return result_dict, soft_unifies