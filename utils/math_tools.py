"""
Math computation helpers built on sympy.
Covers the common operations we need: evaluate, solve, differentiate, integrate, limits, etc.
"""

import re
import math
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def safe_eval(expression):
    """Evaluate a math expression. Returns dict with result + steps."""
    steps = []
    try:
        expr = expression.strip()
        steps.append(f"input: {expr}")

        # try sympy symbolic first
        try:
            sym = parse_expr(expr, transformations=TRANSFORMS)
            steps.append(f"parsed: {sym}")
            simplified = sp.simplify(sym)
            steps.append(f"simplified: {simplified}")
            num = float(simplified.evalf())
            steps.append(f"value: {num}")
            return {"success": True, "symbolic": str(simplified), "numerical": num, "steps": steps}
        except Exception:
            pass

        # fallback: restricted eval for simple arithmetic
        allowed = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "pi": math.pi, "e": math.e, "exp": math.exp,
            "factorial": math.factorial, "pow": pow,
        }
        result = eval(expr, {"__builtins__": {}}, allowed)
        steps.append(f"result: {result}")
        return {"success": True, "symbolic": str(result), "numerical": float(result), "steps": steps}

    except Exception as e:
        return {"success": False, "error": str(e), "steps": steps}


def solve_equation(equation_str, variable="x"):
    steps = []
    try:
        var = sp.Symbol(variable)
        steps.append(f"solving for {variable}")

        if "=" in equation_str:
            parts = equation_str.split("=")
            lhs = parse_expr(parts[0].strip(), transformations=TRANSFORMS)
            rhs = parse_expr(parts[1].strip(), transformations=TRANSFORMS)
            eq = sp.Eq(lhs, rhs)
        else:
            eq = parse_expr(equation_str, transformations=TRANSFORMS)

        steps.append(f"equation: {eq}")
        solutions = sp.solve(eq, var)
        steps.append(f"solutions: {solutions}")
        return {"success": True, "solutions": [str(s) for s in solutions], "steps": steps}

    except Exception as e:
        return {"success": False, "error": str(e), "steps": steps}


def compute_derivative(expr_str, variable="x"):
    steps = []
    try:
        var = sp.Symbol(variable)
        expr = parse_expr(expr_str, transformations=TRANSFORMS)
        steps.append(f"d/d{variable} of {expr}")
        deriv = sp.simplify(sp.diff(expr, var))
        steps.append(f"result: {deriv}")
        return {"success": True, "derivative": str(deriv), "steps": steps}
    except Exception as e:
        return {"success": False, "error": str(e), "steps": steps}


def compute_integral(expr_str, variable="x"):
    steps = []
    try:
        var = sp.Symbol(variable)
        expr = parse_expr(expr_str, transformations=TRANSFORMS)
        steps.append(f"integrating {expr} w.r.t. {variable}")
        result = sp.integrate(expr, var)
        steps.append(f"result: {result} + C")
        return {"success": True, "integral": str(result) + " + C", "steps": steps}
    except Exception as e:
        return {"success": False, "error": str(e), "steps": steps}


def compute_limit(expr_str, variable="x", point="0"):
    steps = []
    try:
        var = sp.Symbol(variable)
        expr = parse_expr(expr_str, transformations=TRANSFORMS)
        pt = parse_expr(point, transformations=TRANSFORMS)
        steps.append(f"limit of {expr} as {variable} -> {point}")
        result = sp.limit(expr, var, pt)
        steps.append(f"result: {result}")
        return {"success": True, "limit": str(result), "steps": steps}
    except Exception as e:
        return {"success": False, "error": str(e), "steps": steps}


def compute_matrix_op(operation, matrix_data):
    steps = []
    try:
        M = sp.Matrix(matrix_data)
        steps.append(f"matrix: {M}")

        if operation == "determinant":
            det = M.det()
            steps.append(f"det = {det}")
            return {"success": True, "result": str(det), "steps": steps}
        elif operation == "inverse":
            inv = M.inv()
            steps.append(f"inverse = {inv}")
            return {"success": True, "result": str(inv), "steps": steps}
        elif operation == "eigenvalues":
            ev = M.eigenvals()
            steps.append(f"eigenvalues = {ev}")
            return {"success": True, "result": str(ev), "steps": steps}
        else:
            return {"success": False, "error": f"unknown op: {operation}", "steps": steps}

    except Exception as e:
        return {"success": False, "error": str(e), "steps": steps}


def compute_probability(expression):
    """Handle nCr/nPr style probability expressions."""
    steps = []
    try:
        expr = expression.strip()
        steps.append(f"probability expr: {expr}")

        # convert common notations
        expr = re.sub(r'(\d+)C(\d+)', r'binomial(\1,\2)', expr)
        expr = re.sub(r'(\d+)P(\d+)', r'factorial(\1)/factorial(\1-\2)', expr)

        result = parse_expr(
            expr, transformations=TRANSFORMS,
            local_dict={"binomial": sp.binomial, "factorial": sp.factorial}
        )
        val = result.evalf()
        steps.append(f"result: {val}")
        return {"success": True, "result": str(val), "symbolic": str(result), "steps": steps}
    except Exception as e:
        return {"success": False, "error": str(e), "steps": steps}
