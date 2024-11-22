class NotFittedError(ValueError, AttributeError):
    pass


class QMLabWarning(UserWarning):
    def __init__(self, *msg):
        super().__init__(" ".join(msg))
        self.msg = " ".join(msg)

    def __str__(self):
        return repr(self.msg)
