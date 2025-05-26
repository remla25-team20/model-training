from pylint.checkers import BaseChecker
from pylint.lint import PyLinter
import astroid

class HardcodedHyperparamChecker(BaseChecker):
    name = "hardcoded-hyperparam-checker"
    msgs = {
        "C9001": (
            "Hardcoded hyperparameter %s detected. Use config or argument instead.",
            "hardcoded-hyperparam",
            "Avoid hardcoding ML hyperparameters directly in function calls.",
        ),
    }

    def visit_call(self, node: astroid.Call):
        for keyword in node.keywords:
            if isinstance(keyword.value, astroid.Const):
                if isinstance(keyword.value.value, float):
                    self.add_message(
                        "hardcoded-hyperparam", node=keyword, args=(keyword.value.value,)
                    )

def register(linter: PyLinter) -> None:
    linter.register_checker(HardcodedHyperparamChecker(linter))
