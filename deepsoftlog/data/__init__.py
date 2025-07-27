from deepsoftlog.algebraic_prover.terms.expression import Constant, Expr, Fact
from deepsoftlog.algebraic_prover.terms.variable import Variable

from .query import Query
from deepsoftlog.logic.soft_term import SoftTerm, TensorTerm
import csv

def load_csv_file(filename: str):
    with open(filename, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader]

def load_tsv_file(filename: str):
    with open(filename, "r") as f:
        return [line.strip().split("\t") for line in f.readlines()]


def data_to_prolog(rows, name="r", **kwargs):
    for row in rows:
        args = [Constant(a) for a in row]
        args = [args[1], args[0], args[2]]
        args = [SoftTerm(a) for a in args]
        yield Query(Expr(name, *args), **kwargs)

def sg_to_prolog(dataset_instance, name="scene_graph"):
    # yield Fact(Expr("groundtruth", SoftTerm(Constant("man")), Constant("bbox1")))
    # yield Fact(Expr("object", SoftTerm(Constant("man")), Constant("bbox1")))
    scene_graph = dataset_instance.scene_graph
   
    # Relationship facts
    for row in scene_graph.triplets:
        args = [Constant(a) for a in row]
        args = [SoftTerm(args[1]), args[0], args[2]]
        yield Fact(Expr(name, *args))

    # Object facts
    for bbox_id, (obj_name, _) in scene_graph.bounding_boxes.items():
        args = [SoftTerm(Constant(obj_name)), Constant(bbox_id)]
        yield Fact(Expr("object", *args))

    # Groundtruth fact
    print(f"Groundtruth: {dataset_instance.target[0]}, {dataset_instance.target[1]}")
    args = [SoftTerm(Constant(dataset_instance.target[0])), Constant(dataset_instance.target[1])]
    yield Fact(Expr("groundtruth", SoftTerm(Constant(dataset_instance.target[0])), Constant(dataset_instance.target[1])))
   


def ontology_to_prolog(rows, name="ontology"):
    for row in rows:
        args = [Constant(a) for a in row]
        args = [args[1], args[0], args[2]]
        args = [SoftTerm(a) for a in args]
        yield Expr(name, *args)
         

def expression_to_prolog(query, name="expression", **kwargs):
    X = Variable("X")
      
    target = Expr("target", X)  # Will match target(X)
    type_expr = Expr("type", X, SoftTerm(Constant("person"))) # Will match type(X, Y)
    ontology_expr = Expr("ontology", SoftTerm(Constant("hyponym")), X, SoftTerm(Constant("person")))
    groundtruth = Expr("groundtruth", Constant("man"), Constant("bbox1"))
    simple = Expr("target", Constant("man"))
    
    expression = Expr("expression", SoftTerm(Constant("near")), SoftTerm(Constant("person")), SoftTerm(Constant("woman")))
    
    combined_expr = target & type_expr & ontology_expr
    print(f"Combined expression: {combined_expr}")
    return Query(combined_expr)
    return Query(type_expr)





# def query_to_prolog(string, **kwargs):
#     # Parse the input string into individual expressions
#     expressions = []
    
#     # Split by commas and clean up whitespace, but be careful with nested commas inside parentheses
#     parts = []
#     current_part = ""
#     paren_count = 0
    
#     for char in string:
#         if char == '(' or char == '[' or char == '{':
#             paren_count += 1
#             current_part += char
#         elif char == ')' or char == ']' or char == '}':
#             paren_count -= 1
#             current_part += char
#         elif char == ',' and paren_count == 0:
#             parts.append(current_part.strip())
#             current_part = ""
#         else:
#             current_part += char
    
#     if current_part.strip():
#         parts.append(current_part.strip())
    
#     # Parse each expression
#     target_var = None
#     for expr_str in parts:
#         if expr_str.startswith("target("):
#             # Extract the variable
#             var_name = expr_str[len("target("):-1].strip()
#             target_var = Variable(var_name)
#             expressions.append(Expr("target", target_var))
        
#         elif expr_str.startswith("type("):
#             # Extract the variable and type
#             content = expr_str[len("type("):-1].strip()
#             # Use split with maxsplit to handle potential commas in the type name
#             var_parts = content.split(',', 1)
#             if len(var_parts) < 2:
#                 raise ValueError(f"Invalid type expression: {expr_str}")
                
#             var_name = var_parts[0].strip()
#             type_name = var_parts[1].strip()
            
#             # Remove quotes if present
#             if type_name.startswith("'") and type_name.endswith("'"):
#                 type_name = type_name[1:-1]
#             if type_name.startswith('"') and type_name.endswith('"'):
#                 type_name = type_name[1:-1]
            
#             # Ensure we're referencing the same variable
#             var = target_var if target_var and var_name == target_var.name else Variable(var_name)
#             expressions.append(Expr("type", var, SoftTerm(Constant(type_name))))
        
#         elif expr_str.startswith("expression("):
#             # Extract predicate, var and object
#             content = expr_str[len("expression("):-1].strip()
            
#             # Split carefully to handle potential commas in the arguments
#             expr_parts = []
#             part = ""
#             nested_count = 0
            
#             for char in content:
#                 if char == '(' or char == '[' or char == '{':
#                     nested_count += 1
#                     part += char
#                 elif char == ')' or char == ']' or char == '}':
#                     nested_count -= 1
#                     part += char
#                 elif char == ',' and nested_count == 0:
#                     expr_parts.append(part.strip())
#                     part = ""
#                 else:
#                     part += char
            
#             if part.strip():
#                 expr_parts.append(part.strip())
            
#             if len(expr_parts) < 3:
#                 raise ValueError(f"Invalid expression: {expr_str}")
            
#             predicate = expr_parts[0].strip()
#             var_name = expr_parts[1].strip()
#             obj_name = expr_parts[2].strip()
            
#             # Handle negation if present
#             negated = False
#             if predicate.startswith("~"):
#                 negated = True
#                 predicate = predicate[1:]
            
#             # Remove quotes if present
#             if obj_name.startswith("'") and obj_name.endswith("'"):
#                 obj_name = obj_name[1:-1]
#             if obj_name.startswith('"') and obj_name.endswith('"'):
#                 obj_name = obj_name[1:-1]
            
#             # Create the expression
#             var = target_var if target_var and var_name == target_var.name else SoftTerm(Constant(var_name))
#             expr = Expr("expression", SoftTerm(Constant(predicate)), var, SoftTerm(Constant(obj_name)))
            
#             # Apply negation if needed
#             if negated:
#                 expr = ~expr
            
#             expressions.append(expr)
    
#     # Combine all expressions with the & operator
#     if not expressions:
#         return Query(None)  # Handle empty case
    
#     combined_expr = expressions[0]
#     for expr in expressions[1:]:
#         combined_expr = combined_expr & expr
    
#     return Query(combined_expr)


def query_to_prolog(string, **kwargs):
    # Parse the input string into individual expressions
    expressions = []
    
    # Split by commas and clean up whitespace, but be careful with nested commas inside parentheses
    parts = []
    current_part = ""
    paren_count = 0
    
    for char in string:
        if char == '(' or char == '[' or char == '{':
            paren_count += 1
            current_part += char
        elif char == ')' or char == ']' or char == '}':
            paren_count -= 1
            current_part += char
        elif char == ',' and paren_count == 0:
            parts.append(current_part.strip())
            current_part = ""
        else:
            current_part += char
    
    if current_part.strip():
        parts.append(current_part.strip())
    
    # Parse each expression
    target_var = None
    for expr_str in parts:
        if expr_str.startswith("target("):
            # Extract the variable
            var_name = expr_str[len("target("):-1].strip()
            target_var = Variable(var_name)
            expressions.append(Expr("target", target_var))
        
        elif expr_str.startswith("type("):
            # Extract the variable and type
            content = expr_str[len("type("):-1].strip()
            # Use split with maxsplit to handle potential commas in the type name
            var_parts = content.split(',', 1)
            if len(var_parts) < 2:
                raise ValueError(f"Invalid type expression: {expr_str}")
                
            var_name = var_parts[0].strip()
            type_name = var_parts[1].strip()
            
            # Remove quotes if present
            if type_name.startswith("'") and type_name.endswith("'"):
                type_name = type_name[1:-1]
            if type_name.startswith('"') and type_name.endswith('"'):
                type_name = type_name[1:-1]
            
            # Ensure we're referencing the same variable
            var = target_var if target_var and var_name == target_var.name else Variable(var_name)
            expressions.append(Expr("type", var, SoftTerm(Constant(type_name))))
        
        elif expr_str.startswith("expression("):
            # Extract predicate, subject and object
            content = expr_str[len("expression("):-1].strip()
            
            # Split carefully to handle potential commas in the arguments
            expr_parts = []
            part = ""
            nested_count = 0
            
            for char in content:
                if char == '(' or char == '[' or char == '{':
                    nested_count += 1
                    part += char
                elif char == ')' or char == ']' or char == '}':
                    nested_count -= 1
                    part += char
                elif char == ',' and nested_count == 0:
                    expr_parts.append(part.strip())
                    part = ""
                else:
                    part += char
            
            if part.strip():
                expr_parts.append(part.strip())
            
            if len(expr_parts) < 3:
                raise ValueError(f"Invalid expression: {expr_str}")
            
            predicate = expr_parts[0].strip()
            subject_name = expr_parts[1].strip()
            object_name = expr_parts[2].strip()
            
            # Handle negation if present
            negated = False
            if predicate.startswith("~"):
                negated = True
                predicate = predicate[1:]
            
            # Remove quotes if present in object name
            if object_name.startswith("'") and object_name.endswith("'"):
                object_name = object_name[1:-1]
            if object_name.startswith('"') and object_name.endswith('"'):
                object_name = object_name[1:-1]
            
            # Determine if subject is a variable or constant
            if subject_name.startswith('X') or subject_name.startswith('Y') or subject_name.startswith('Z'):
                # Subject is a variable
                subject = target_var if target_var and subject_name == target_var.name else Variable(subject_name)
            else:
                # Subject is a constant that should be soft
                subject = SoftTerm(Constant(subject_name))
                
            # Determine if object is a variable or constant
            if object_name.startswith('X') or object_name.startswith('Y') or object_name.startswith('Z'):
                # Object is a variable
                object_term = target_var if target_var and object_name == target_var.name else Variable(object_name)
            else:
                # Object is a constant that should be soft
                object_term = SoftTerm(Constant(object_name))
            
            # Create the expression with properly handled variables
            expr = Expr("expression", SoftTerm(Constant(predicate)), subject, object_term)
            
            # Apply negation if needed
            if negated:
                expr = ~expr
            
            expressions.append(expr)
    
    # Combine all expressions with the & operator
    if not expressions:
        return Query(None)  # Handle empty case
    
    combined_expr = expressions[0]
    for expr in expressions[1:]:
        combined_expr = combined_expr & expr
    
    return Query(combined_expr)

        
def to_prolog_image(img):
    return SoftTerm(Expr("lenet5", TensorTerm(img)))
