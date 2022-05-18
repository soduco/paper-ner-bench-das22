import util.config as config


class ReproducibilityError(AssertionError):
    def __init__(self, actual, expected, rule=None):
        self.message = f"Something violated the reproducibility rule. Expected {expected} but got {actual}"
        super().__init__(self.message)


def assert_expected(actual, expected, rule=None):
    if config.AS_IN_THE_PAPER:
        try:
            if rule:
                assert rule(actual, expected)
            else:
                assert actual == expected
        except AssertionError as ae:
            raise ReproducibilityError(actual, expected)
