import sympy as sp
import param
from sympy import Tuple
from zoomy_core.misc.misc import Zstruct, ZArray


def listify(expr):
    if isinstance(expr, (ZArray, sp.NDimArray)):
        return Tuple(*expr.tolist())
    elif hasattr(expr, "args") and expr.args:
        return expr.func(*[listify(a) for a in expr.args])
    else:
        return expr


class Function(param.Parameterized):
    name = param.String(default="Function")
    args = param.ClassSelector(class_=Zstruct, default=None)
    definition = param.Parameter(default=None)

    def __init__(self, **params):
        super().__init__(**params)
        if self.args is None:
            self.args = Zstruct()
        if self.definition is None:
            self.definition = ZArray([0])


class SymbolicRegistrar:
    def register_symbolic_function(self, name, method_ref, sig_struct):
        definition = method_ref()
        self.functions[name] = Function(
            name=name, definition=definition, args=sig_struct
        )

        def proxy_caller(*input_args):
            if not input_args:
                return self.functions[name].definition

            final_args = []
            for arg in input_args:
                if hasattr(arg, "_symbolic_name") and arg._symbolic_name:
                    final_args.append(sp.Symbol(arg._symbolic_name))
                else:
                    if hasattr(arg, "get_list"):
                        final_args.extend(arg.get_list())
                    elif isinstance(arg, (list, tuple)):
                        final_args.extend(arg)
                    else:
                        final_args.append(arg)

            prefix = "Model<T>::" if self.name == "Model" else ""
            full_name = f"{prefix}{name}"

            shape = getattr(definition, "shape", ())
            main_call = sp.Function(full_name)(*final_args)

            if not shape or shape == (1,):
                return main_call

            size = 1
            for s in shape:
                size *= s
            base = sp.IndexedBase(main_call, shape=(size,))
            indexed_elements = [base[i] for i in range(size)]
            return ZArray(indexed_elements).reshape(*shape)

        self.call[name] = proxy_caller
