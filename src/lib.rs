use pyo3::prelude::*;

#[allow(nonstandard_style)]
#[pymodule]
mod typed_numpy
{
    use egglog::ast::{Action, Command, Expr, Fact, Schema};
    use egglog::prelude::*;
    use egglog::add_primitive;
    use egglog::sort::{F, OrderedFloat};
    use pyo3::prelude::*;

    #[pyclass]
    enum RustExpr
    {
        Constant { value : f64 },
        Tensor { dims : Vec<String> },
        Exp { x : Py<RustExpr> },
        Sin { x : Py<RustExpr> },
        Cos { x : Py<RustExpr> },
        Sqrt { x : Py<RustExpr> },
        Add { x : Py<RustExpr>, y : Py<RustExpr> },
        Sub { x : Py<RustExpr>, y : Py<RustExpr> },
        Mul { x : Py<RustExpr>, y : Py<RustExpr> },
        Div { x : Py<RustExpr>, y : Py<RustExpr> },
        BinaryMax { x : Py<RustExpr>, y : Py<RustExpr> },
        Repeat { dim : String, x : Py<RustExpr> },
        Sum { dim : String, start : i64, end : i64, x : Py<RustExpr> },
        Max { dim : String, start : i64, end : i64, x : Py<RustExpr> },
    }

    fn RustExprToEgglogExpr(x : &Py<RustExpr>) -> Expr
    {
        match &x.get()
        {
            RustExpr::Constant { value } =>
            {
                return exprs::call("Constant", vec![exprs::float(*value)]);
            }
            RustExpr::Tensor { dims } =>
            {
                let dim_exprs : Vec<Expr> = dims.iter().map(|d| exprs::string(d)).collect();
                let dims_set = exprs::call("set-of", dim_exprs);
                return exprs::call("Tensor", vec![dims_set]);
            }
            RustExpr::Exp { x } =>
            {
                let child = RustExprToEgglogExpr(&x);
                return exprs::call("Exp", vec![child]);
            }
            RustExpr::Sin { x } =>
            {
                let child = RustExprToEgglogExpr(&x);
                return exprs::call("Sin", vec![child]);
            }
            RustExpr::Cos { x } =>
            {
                let child = RustExprToEgglogExpr(&x);
                return exprs::call("Cos", vec![child]);
            }
            RustExpr::Sqrt { x } =>
            {
                let child = RustExprToEgglogExpr(&x);
                return exprs::call("Sqrt", vec![child]);
            }
            RustExpr::Add { x, y } =>
            {
                let lhs = RustExprToEgglogExpr(&x);
                let rhs = RustExprToEgglogExpr(&y);
                return exprs::call("Add", vec![lhs, rhs]);
            }
            RustExpr::Sub { x, y } =>
            {
                let lhs = RustExprToEgglogExpr(&x);
                let rhs = RustExprToEgglogExpr(&y);
                return exprs::call("Sub", vec![lhs, rhs]);
            }
            RustExpr::Mul { x, y } =>
            {
                let lhs = RustExprToEgglogExpr(&x);
                let rhs = RustExprToEgglogExpr(&y);
                return exprs::call("Mul", vec![lhs, rhs]);
            }
            RustExpr::Div { x, y } =>
            {
                let lhs = RustExprToEgglogExpr(&x);
                let rhs = RustExprToEgglogExpr(&y);
                return exprs::call("Div", vec![lhs, rhs]);
            }
            RustExpr::BinaryMax { x, y } =>
            {
                let lhs = RustExprToEgglogExpr(&x);
                let rhs = RustExprToEgglogExpr(&y);
                return exprs::call("BinaryMax", vec![lhs, rhs]);
            }
            RustExpr::Repeat { dim, x } =>
            {
                let child = RustExprToEgglogExpr(&x);
                return exprs::call("Repeat", vec![exprs::string(dim), child]);
            }
            RustExpr::Sum { dim, start, end, x } =>
            {
                let child = RustExprToEgglogExpr(&x);
                return exprs::call("Sum", vec![exprs::string(dim), exprs::int(*start), exprs::int(*end), child]);
            }
            RustExpr::Max { dim, start, end, x } =>
            {
                let child = RustExprToEgglogExpr(&x);
                return exprs::call("Max", vec![exprs::string(dim), exprs::int(*start), exprs::int(*end), child]);
            }
        };
    }

    #[derive(Debug, Clone)]
    struct TypedNumpyError
    {
        message : String,
    }

    impl From<egglog::Error> for TypedNumpyError
    {
        fn from(err : egglog::Error) -> TypedNumpyError
        {
            return TypedNumpyError
            {
                message : err.to_string(),
            };
        }
    }

    impl From<TypedNumpyError> for PyErr
    {
        fn from(err : TypedNumpyError) -> PyErr
        {
            return pyo3::exceptions::PyRuntimeError::new_err(err.message);
        }
    }

    #[pyclass]
    struct RustEgraph
    {
        eg : egglog::EGraph,
        current_id : i64,
    }

    static Ruleset : &str = "rewrites";

    fn NewRustEgraph() -> Result<egglog::EGraph, egglog::Error>
    {
        let mut eg = egglog::EGraph::default();
        eg.declare_sort(
            "StringSet".to_string(),
            &Some(
                ("Set".to_string(),
                vec![Expr::Var(span!(), "String".to_string())]),
            ),
            span!(),
        )?;
            
        datatype!(&mut eg,
            (datatype Expr
                (Constant f64)
                (Tensor StringSet)
                (Exp Expr)
                (Sin Expr)
                (Cos Expr)
                (Sqrt Expr)
                (Add Expr Expr)
                (Sub Expr Expr)
                (Mul Expr Expr)
                (Div Expr Expr)
                (BinaryMax Expr Expr)
                (Repeat String Expr)
                (Sum String i64 i64 Expr)
                (Max String i64 i64 Expr)
            )
        );

        add_primitive!(&mut eg, "exp" = |a: F| -> F { F::from(OrderedFloat((**a).exp())) });
        add_primitive!(&mut eg, "sin" = |a: F| -> F { F::from(OrderedFloat((**a).sin())) });
        add_primitive!(&mut eg, "cos" = |a: F| -> F { F::from(OrderedFloat((**a).cos())) });
        add_primitive!(&mut eg, "sqrt" = |a: F| -> F { F::from(OrderedFloat((**a).sqrt())) });

        add_ruleset(&mut eg, Ruleset)?;

        macro_rules! add_rule
        {
            ($lhs:tt => $rhs:tt) =>
            {
                rule(&mut eg, Ruleset, facts![(= x $lhs)], actions![(union x $rhs)])?;
            };
            ($lhs:tt <=> $rhs:tt) =>
            {
                add_rule!($lhs => $rhs);
                add_rule!($rhs => $lhs);
            };
        }

        add_rule!((Add a b) => (Add b a)); // commutativity
        add_rule!((Mul a b) => (Mul b a)); // commutativity
        add_rule!((Add (Add a b) c) <=> (Add a (Add b c))); // associativity
        add_rule!((Mul (Mul a b) c) <=> (Mul a (Mul b c))); // associativity
        add_rule!((Mul a (Add b c)) <=> (Add (Mul a b) (Mul a c))); // distributivity
        add_rule!((Div (Add a b) c) <=> (Add (Div a c) (Div b c))); // div_distributivity
        // repeat_over_unary_ops
        add_rule!((Exp (Repeat D a)) <=> (Repeat D (Exp a)));
        add_rule!((Sin (Repeat D a)) <=> (Repeat D (Sin a)));
        add_rule!((Cos (Repeat D a)) <=> (Repeat D (Cos a)));
        add_rule!((Sqrt (Repeat D a)) <=> (Repeat D (Sqrt a)));
        // repeat_over_binary_ops
        add_rule!((Add (Repeat D a) (Repeat D b)) <=> (Repeat D (Add a b)));
        add_rule!((Sub (Repeat D a) (Repeat D b)) <=> (Repeat D (Sub a b)));
        add_rule!((Mul (Repeat D a) (Repeat D b)) <=> (Repeat D (Mul a b)));
        add_rule!((Div (Repeat D a) (Repeat D b)) <=> (Repeat D (Div a b)));
        add_rule!((BinaryMax (Repeat D a) (Repeat D b)) <=> (Repeat D (BinaryMax a b)));
        // reorder_reductions
        add_rule!((Sum D s e (Sum E m n a)) => (Sum E m n (Sum D s e a)));
        add_rule!((Max D s e (Max E m n a)) => (Max E m n (Max D s e a)));
        // combine_reductions
        add_rule!((Add (Sum D s m x) (Sum D m e x)) => (Sum D s e x));
        add_rule!((BinaryMax (Max D s m x) (Max D m e x)) => (Max D s e x));

        add_rule!((Repeat D (Repeat E x)) => (Repeat E (Repeat D x))); // reorder_repeats
        add_rule!((Sum D s e (Div a (Repeat D b))) <=> (Div (Sum D s e a) b)); // factor_out_of_sum
        add_rule!((Mul x (Constant (unquote exprs::float(0.0)))) => (Constant (unquote exprs::float(0.0)))); // mul_by_zero
        add_rule!((Div a (Div b c)) <=> (Mul a (Div c b))); // div_of_quotient_to_mul_of_reciprocal
        add_rule!((Add (Div a d) (Div b d)) <=> (Div (Add a b) d)); // add_fractions
        add_rule!((Sub (Div a d) (Div b d)) <=> (Div (Sub a b) d)); // sub_fractions
        add_rule!((Mul (Div a c) (Div b d)) <=> (Div (Mul a b) (Mul c d))); // multiply_fractions
        add_rule!((Div (Mul a b) (Mul c d)) <=> (Mul (Div a c) (Div b d))); // decompose_fractions
        add_rule!((Sub a a) => (Constant (unquote exprs::float(0.0)))); // sub_by_self
        add_rule!((Div a a) => (Constant (unquote exprs::float(1.0)))); // div_by_self
        add_rule!((Add a (Constant (unquote exprs::float(0.0)))) => a); // identity_add
        add_rule!((Sub a (Constant (unquote exprs::float(0.0)))) => a); // identity_sub
        add_rule!((Mul a (Constant (unquote exprs::float(1.0)))) => a); // identity_mul
        add_rule!((Div a (Constant (unquote exprs::float(1.0)))) => a); // identity_div
        add_rule!((Exp (Constant a)) => (Constant (exp a))); // constant_folding_exp
        add_rule!((Sin (Constant a)) => (Constant (sin a))); // constant_folding_sin
        add_rule!((Cos (Constant a)) => (Constant (cos a))); // constant_folding_cos
        add_rule!((Sqrt (Constant a)) => (Constant (sqrt a))); // constant_folding_sqrt
        add_rule!((Add (Constant a) (Constant b)) => (Constant (+ a b))); // constant_folding_add
        add_rule!((Sub (Constant a) (Constant b)) => (Constant (- a b))); // constant_folding_sub
        add_rule!((Mul (Constant a) (Constant b)) => (Constant (* a b))); // constant_folding_mul
        add_rule!((Div (Constant a) (Constant b)) => (Constant (/ a b))); // constant_folding_div
        add_rule!((Exp (Add x y)) <=> (Mul (Exp x) (Exp y))); // product_of_exp
        add_rule!((Exp (Sub x y)) <=> (Div (Exp x) (Exp y))); // quotient_of_exp

        // log_sum_exp_stability
        add_rule!((Sum D s e (Exp x)) <=>
            (Mul
                (Exp (Max D s e x))
                (Sum D s e
                    (Exp
                        (Sub x
                            (Repeat D
                                (Max D s e x)))))));

        // numerically_stable_softmax
        add_rule!((Div (Exp x) (Repeat D (Sum D s e (Exp x)))) <=>
            (Div (Exp (Sub x (Repeat D (Max D s e x))))
                (Repeat D (Sum D s e (Exp (Sub x (Repeat D (Max D s e x))))))));

        return Ok(eg);
    }

    impl RustEgraph
    {
        fn next_id(&mut self) -> i64
        {
            let result = self.current_id;
            self.current_id += 1;
            return result;
        }
    }

    #[pymethods]
    impl RustEgraph
    {
        #[new]
        fn py_new() -> PyResult<Self>
        {
            let eg = NewRustEgraph().map_err(TypedNumpyError::from)?;
            return Ok(
                RustEgraph
                {
                    eg : eg,
                    current_id : 0,
                }
            );
        }

        fn insert_expression(&mut self, expr : Py<RustExpr>) -> PyResult<String>
        {
            let egglog_expr = RustExprToEgglogExpr(&expr);
            let expr_id = format!("expr_{}", self.next_id());
            self.eg.run_program(
                vec![
                    Command::Action(
                        Action::Let(span!(), expr_id.clone(), egglog_expr)
                    ),
                ],
            ).map_err(TypedNumpyError::from)?;
            return Ok(expr_id);
        }

        #[pyo3(signature = (a_id, b_id, verbose=false))]
        fn incrementally_check_equivalence(
            &mut self,
            a_id : &str,
            b_id : &str,
            verbose : bool,
        ) -> PyResult<bool>
        {
            let verbose = verbose || std::env::var("EGGLOG_VERBOSE").is_ok();
            for i in 0..10
            {
                if verbose
                {
                    println!("Running ruleset iteration {}", i);
                }
                let run_result = run_ruleset(
                    &mut self.eg,
                    Ruleset,
                ).map_err(TypedNumpyError::from)?;
                if verbose
                {
                    println!("Ruleset run result: {:#?}", run_result);
                }

                // Construct the equality fact directly
                let equality_fact = Fact::Eq(
                    span!(),
                    exprs::var(a_id),
                    exprs::var(b_id)
                );

                let check_result = self.eg.run_program(
                    vec![
                        Command::PrintOverallStatistics,
                        Command::Check(
                            span!(),
                            vec![equality_fact],
                        )
                    ],
                );
                if verbose
                {
                    println!("Check result: {:#?}", check_result);
                }
                if check_result.is_ok()
                {
                    if verbose
                    {
                        println!("Found equivalence at iteration {}", i);
                    }
                    return Ok(true);
                }
            }
            if verbose
            {
                println!("No equivalence found after 10 iterations");
            }
            return Ok(false);
        }
    }
}
