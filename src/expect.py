"Constrained-decoding states for a function-call JSON document."

from abc import ABC, abstractmethod

from .data_models import StateContext


class Expect(ABC):
    "Base state with shared prefix matching behavior."

    def __init__(self, context: StateContext):
        super().__init__()
        self.context = context
        self.tokens: list[int] = []

    def continuations(self, choices: tuple[tuple[int, ...], ...]) -> set[int]:
        "Find the next token allowed by every matching choice."
        if not self.tokens:
            return {choice[0] for choice in choices if choice}

        allowed: set[int] = set()
        consumed = len(self.tokens)
        for choice in choices:
            if consumed >= len(choice):
                continue
            if all(want == got for want, got in zip(choice, self.tokens)):
                allowed.add(choice[consumed])
        return allowed

    @abstractmethod
    def get_allowed(self) -> set[int]:
        "Return tokens permitted in this segment."

    @abstractmethod
    def next_state(self) -> "Expect | None":
        "Advance to the state that owns the next JSON segment."


class ExpectEnd(Expect):
    def get_allowed(self) -> set[int]:
        return self.continuations((self.context.end,))

    def next_state(self) -> None:
        return None


class ExpectKVSep(Expect):
    def get_allowed(self) -> set[int]:
        return self.continuations((self.context.kvsep,))

    def next_state(self) -> Expect:
        return ExpectValue(self.context)


class ExpectKey(Expect):
    def get_allowed(self) -> set[int]:
        if not self.context.parameters:
            return set()
        return self.continuations((self.context.parameters[0],))

    def next_state(self) -> Expect:
        if self.context.parameters:
            self.context.parameters.pop(0)
        return ExpectKVSep(self.context)


class ExpectSep(Expect):
    def get_allowed(self) -> set[int]:
        return self.continuations((self.context.sep,))

    def next_state(self) -> Expect:
        return ExpectKey(self.context)


class ExpectValue(Expect):
    def get_allowed(self) -> set[int]:
        # Negative IDs allow arbitrary JSON string tokens.
        return {-1, -2}

    def next_state(self) -> Expect:
        if self.context.parameters:
            return ExpectSep(self.context)
        return ExpectEnd(self.context)


class ExpectParameters(Expect):
    def get_allowed(self) -> set[int]:
        return self.continuations((self.context.param_def,))

    def next_state(self) -> Expect:
        if self.context.parameters:
            return ExpectKey(self.context)
        return ExpectEnd(self.context)


class ExpectFunction(Expect):
    def get_allowed(self) -> set[int]:
        return self.continuations(tuple(self.context.functions))

    def next_state(self) -> Expect:
        selected = tuple(self.tokens)
        self.context.parameters = list(self.context.functions[selected])
        return ExpectParameters(self.context)
