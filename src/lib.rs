use pyo3::prelude::*;

#[allow(nonstandard_style)]
#[pymodule]
mod typed_numpy
{
    use egglog::ast::Schema;
    use egglog::prelude::*;
    use pyo3::prelude::*;

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
    struct Egraph
    {
        eg : egglog::EGraph,
    }

    fn NewEgraph() -> Result<egglog::EGraph, egglog::Error>
    {
        let mut eg = egglog::EGraph::default();
        datatype!(&mut eg,
            (datatype Expr
                (Constant f64)
                (Tensor)
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
        let ruleset = "rewrites";
        add_ruleset(&mut eg, ruleset)?;

        macro_rules! add_rule
        {
            ($lhs:tt => $rhs:tt) =>
            {
                rule(&mut eg, ruleset, facts![(= x $lhs)], actions![(union x $rhs)])?;
            };
            ($lhs:tt <=> $rhs:tt) =>
            {
                add_rule!($lhs => $rhs);
                add_rule!($rhs => $lhs);
            };
        }

        macro_rules! for_each_unary_op
        {
            ($($rule:tt)*) =>
            {
                macro_rules! __inner
                {
                    ($op:ident) =>
                    {
                        add_rule!($($rule)*);
                    };
                }
                __inner!(Exp);
                __inner!(Sin);
                __inner!(Cos);
                __inner!(Sqrt);
            };
        }

        macro_rules! for_each_binary_op
        {
            ($($rule:tt)*) =>
            {
                macro_rules! __inner
                {
                    ($op:ident) =>
                    {
                        add_rule!($($rule)*);
                    };
                }
                __inner!(Add);
                __inner!(Sub);
                __inner!(Mul);
                __inner!(Div);
                __inner!(BinaryMax);
            };
        }

        macro_rules! for_each_reduction_op
        {
            ($($rule:tt)*) =>
            {
                macro_rules! __inner
                {
                    ($op:ident) =>
                    {
                        add_rule!($($rule)*);
                    };
                }
                __inner!(Add);
                __inner!(Sub);
                __inner!(Mul);
                __inner!(Div);
                __inner!(BinaryMax);
            };
        }

        add_rule!((Add a b) => (Add b a)); // commutativity
        add_rule!((Add (Add a b) c) <=> (Add a (Add b c))); // associativity
        add_rule!((Mul a (Add b c)) <=> (Add (Mul a b) (Mul a c))); // distributivity
        add_rule!((Div (Add a b) c) <=> (Add (Div a c) (Div b c))); // div_distributivity
        for_each_unary_op!(($op (Repeat D s e a)) <=> (Repeat D s e ($op a))); // repeat_over_unary_ops
        for_each_binary_op!(($op (Repeat D s e a) (Repeat D  s e b)) <=> (Repeat D s e ($op a b))) // repeat_over_binary_ops
        for_each_reduction_op!(($op D ($op E a)) <=> ($op E ($op D a))) // reorder_reductions
        add_rule!((Sum D s e (Div a (Repeat D s e b))) <=> (Div (Sum D s e a) b)) // factor_out_of_sum

        return Ok(eg);
    }

    #[pymethods]
    impl Egraph
    {
        #[new]
        fn py_new() -> PyResult<Self>
        {
            let eg = NewEgraph().map_err(TypedNumpyError::from)?;
            return Ok(
                Egraph
                {
                    eg : eg
                }
            );
        }
    }
}
