class QMLabError(Exception):
    """Will get thrown whenever something "general" doesn't
    work in the source code of QMLab.

    Parameters
    ----------
    Exception :
        general Python Exception.
    """

    pass


class InvalidEmbeddingError(ValueError, AttributeError):
    """Will get thrown whenever the user tries to use
    an invalid embedding. An embedding can be invalid
    because
        - it's not implemented yet,
        - it's not a valid PennyLane-Embedding.
    Parameters
    ----------
    ValueError :
    AttributeError :
        Standard Python Errors.
    """

    pass


class NotFittedError(ValueError, AttributeError):
    """Will be thrown when the user tries to
    make a prediction with a model that wasn't
    fitted on data beforehand.

    Parameters
    ----------
    ValueError :
    AttributeError :
        Standard Python Errors.
    """

    pass


class QMLabWarning(UserWarning):
    """Will be raised whenever something "general"
    is ambiguous.

    Parameters
    ----------
    UserWarning :
        Standard Python Error.
    """

    def __init__(self, *msg):
        super().__init__(" ".join(msg))
        self.msg = " ".join(msg)

    def __str__(self):
        return repr(self.msg)


class PerformanceWarning(Warning):
    """Will be raised whenever a model
    exceeds a certain time or computational
    threshold.

    Parameters
    ----------
    Warning :
        Standard Python Warning.
    """

    pass
