"""
Custom Exceptions for IntelliQuery AI

Provides specific exception types for better error handling.
"""


class IntelliQueryError(Exception):
    """Base exception for all IntelliQuery errors"""
    pass


class ConfigurationError(IntelliQueryError):
    """Raised when configuration is invalid or missing"""
    pass


class DatabaseError(IntelliQueryError):
    """Raised when database operations fail"""
    pass


class ValidationError(IntelliQueryError):
    """Raised when data validation fails"""
    pass


class ModelError(IntelliQueryError):
    """Raised when ML model operations fail"""
    pass


class EmbeddingError(IntelliQueryError):
    """Raised when embedding generation fails"""
    pass


class QueryError(IntelliQueryError):
    """Raised when query processing fails"""
    pass

# Made with Bob
