"""Custom exceptions for pyvecdb.

Think of these like warning signs at an amusement park -- they tell you
exactly what went wrong so you can fix it!

Example:
    If you try to add a 3D vector to a store that expects 5D vectors,
    you'll get a ``DimensionMismatchError`` telling you the sizes don't match.

"""


class VecDBError(Exception):
    """Base exception for all pyvecdb errors."""


class DimensionMismatchError(VecDBError):
    """Raise when vector dimensions don't match.

    Imagine trying to fit a square peg in a round hole -- if your vector
    has 3 numbers but the store expects 5, that's a dimension mismatch!
    """


class DuplicateIDError(VecDBError):
    """Raise when a vector record with the same ID already exists.

    Like how every student has a unique student ID -- you can't have
    two students with the same number.
    """


class RecordNotFoundError(VecDBError):
    """Raise when a requested vector record is not found.

    Like looking for a book in the library that doesn't exist -- the
    ID you asked for isn't in the store.
    """


class EmptyStoreError(VecDBError):
    """Raise when searching an empty store.

    You can't find the closest pizza shop if there are no pizza shops!
    """


class EmptyVectorError(VecDBError):
    """Raise when a vector has no components.

    A vector needs at least one number -- an empty list isn't a direction!
    """


class PersistenceError(VecDBError):
    """Raise when a saved store can't be read back from disk.

    Like opening your saved game and finding the file missing or
    scrambled -- rather than starting an empty game and pretending all is
    well, we stop and say which file is the problem.
    """
