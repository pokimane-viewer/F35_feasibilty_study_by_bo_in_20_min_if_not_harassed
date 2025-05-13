
def propositional_truth_values():
    """
    ∀p ∈ {T, F} : p represents a proposition that is either true or false.
    
    In propositional logic, truth values are binary values representing whether a 
    proposition is true or false. These fundamental building blocks allow us to 
    construct and evaluate logical expressions.
    
    Real-world applicability: Propositional logic forms the foundation of computer 
    science, digital circuit design, and formal verification. In software engineering, 
    it enables conditional logic, Boolean expressions, and control flow. It's
    essential in database queries, artificial intelligence, and algorithmic decision-
    making across countless applications.
    """
    return {"T": True, "F": False}

def logical_operators():
    """
    ∀p,q ∈ {T, F} : 
        p ∧ q ≡ min(p,q)
        p ∨ q ≡ max(p,q)
        ¬p ≡ 1-p
    
    Logical operators transform truth values according to specific rules, creating 
    compound propositions whose truth depends on their components' truth values.
    
    Real-world applicability: Logical operators enable complex decision-making in 
    programming, allowing systems to evaluate multiple conditions simultaneously. 
    They're used in search algorithms, data filtering, security access controls, 
    and circuit design where combinations of conditions must be evaluated to 
    determine outcomes.
    """
    return {
        "AND": lambda p, q: p and q,
        "OR": lambda p, q: p or q,
        "NOT": lambda p: not p
    }

def truth_tables():
    """
    T(p ∧ q) = {(T,T)→T, (T,F)→F, (F,T)→F, (F,F)→F}
    T(p ∨ q) = {(T,T)→T, (T,F)→T, (F,T)→T, (F,F)→F}
    T(¬p) = {T→F, F→T}
    
    Truth tables exhaustively list all possible combinations of truth values for 
    propositions and the resulting values of compound expressions formed with them.
    
    Real-world applicability: Truth tables serve as fundamental tools for 
    verifying logical equivalence, designing digital circuits, and checking 
    the validity of arguments. They're used in compiler optimization, hardware 
    verification, and protocol analysis to ensure systems behave correctly under 
    all possible input conditions.
    """
    p_values = [True, False]
    q_values = [True, False]
    
    and_table = {(p, q): p and q for p in p_values for q in q_values}
    or_table = {(p, q): p or q for p in p_values for q in q_values}
    not_table = {p: not p for p in p_values}
    
    return {"AND": and_table, "OR": or_table, "NOT": not_table}

def implication_operator():
    """
    ∀p,q ∈ {T, F} : p → q ≡ ¬p ∨ q
    
    The implication operator represents logical consequence, where p → q is only 
    false when p is true and q is false.
    
    Real-world applicability: Implications model cause-effect relationships and 
    conditional reasoning in AI systems, expert systems, and automated theorem 
    proving. They're crucial in formal specifications, program verification, and 
    rule-based systems where conclusions must be drawn from premises according 
    to logical rules.
    """
    p_values = [True, False]
    q_values = [True, False]
    
    implication_table = {(p, q): (not p) or q for p in p_values for q in q_values}
    return implication_table

def tautology_contradiction():
    """
    ∀p ∈ {T, F} : p ∨ ¬p ≡ T (tautology)
    ∀p ∈ {T, F} : p ∧ ¬p ≡ F (contradiction)
    
    Tautologies are propositions that are always true regardless of the truth 
    values of their components. Contradictions are always false.
    
    Real-world applicability: Identifying tautologies and contradictions helps 
    in simplifying logical circuits, optimizing code paths, and finding logical 
    errors in specifications. They're used in formal proofs, consistency checking 
    of requirements, and detecting redundant or impossible conditions in software.
    """
    p_values = [True, False]
    
    tautology_result = all((p or not p) for p in p_values)
    contradiction_result = all(not (p and not p) for p in p_values)
    
    return {"tautology_verified": tautology_result, 
            "contradiction_verified": contradiction_result}
