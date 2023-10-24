from fittrackee.exceptions import GenericException


class InvalidGPXException(GenericException):
    ...


class WorkoutException(GenericException):
    ...


class WorkoutGPXException(GenericException):
    ...

class InvalidFitException(GenericException):
    ...

class WorkoutFitException(GenericException):
    ...
