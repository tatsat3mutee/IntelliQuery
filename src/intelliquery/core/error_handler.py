"""
Error Handling Framework - Enterprise-Grade Exception Management
================================================================
Centralized error handling with proper categorization and safe error responses.

Features:
- Custom exception hierarchy
- Request-scoped error context
- Safe error messages (no internal details leaked)
- Structured logging integration
"""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Standardized error codes for API responses"""
    # Client errors (4xx)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    
    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    LLM_ERROR = "LLM_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


class ErrorCategory(Enum):
    """Categories for error classification"""
    VALIDATION = "validation"
    SECURITY = "security"
    DATABASE = "database"
    ML_MODEL = "ml_model"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL = "internal"


@dataclass
class ErrorContext:
    """Context information for error tracking"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


class IntelliQueryException(Exception):
    """
    Base exception for all IntelliQuery errors.
    Provides structured error information for API responses.
    """
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        http_status: int = 500,
        details: Optional[Dict[str, Any]] = None,
        internal_message: Optional[str] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: User-safe error message (shown in API response)
            error_code: Standardized error code
            category: Error category for classification
            http_status: HTTP status code for response
            details: Additional details safe to show users
            internal_message: Detailed message for logs only (never shown to users)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.http_status = http_status
        self.details = details or {}
        self.internal_message = internal_message
    
    def to_response_dict(self, request_id: str) -> Dict[str, Any]:
        """Convert to safe API response dictionary"""
        return {
            "success": False,
            "error": {
                "code": self.error_code.value,
                "message": self.message,
                "category": self.category.value,
                "details": self.details,
                "request_id": request_id
            }
        }


class ValidationException(IntelliQueryException):
    """Raised when input validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            category=ErrorCategory.VALIDATION,
            http_status=400,
            details={"field": field, **(details or {})}
        )


class FileValidationException(IntelliQueryException):
    """Raised when file validation fails"""
    
    def __init__(self, message: str, filename: Optional[str] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_INPUT,
            category=ErrorCategory.VALIDATION,
            http_status=400,
            details={"filename": filename}
        )


class FileTooLargeException(IntelliQueryException):
    """Raised when uploaded file exceeds size limit"""
    
    def __init__(self, actual_size: int, max_size: int, filename: Optional[str] = None):
        super().__init__(
            message=f"File too large. Maximum size is {max_size / 1024 / 1024:.0f}MB",
            error_code=ErrorCode.FILE_TOO_LARGE,
            category=ErrorCategory.VALIDATION,
            http_status=413,
            details={
                "filename": filename,
                "actual_size_mb": round(actual_size / 1024 / 1024, 2),
                "max_size_mb": round(max_size / 1024 / 1024, 0)
            }
        )


class UnsupportedFileTypeException(IntelliQueryException):
    """Raised when file type is not supported"""
    
    def __init__(self, filename: str, allowed_types: list):
        super().__init__(
            message=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}",
            error_code=ErrorCode.UNSUPPORTED_FILE_TYPE,
            category=ErrorCategory.VALIDATION,
            http_status=415,
            details={"filename": filename, "allowed_types": allowed_types}
        )


class DatabaseException(IntelliQueryException):
    """Raised when database operations fail"""
    
    def __init__(self, message: str = "Database operation failed", internal_message: Optional[str] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_ERROR,
            category=ErrorCategory.DATABASE,
            http_status=500,
            internal_message=internal_message
        )


class ModelException(IntelliQueryException):
    """Raised when ML model operations fail"""
    
    def __init__(self, message: str = "Model operation failed", internal_message: Optional[str] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_ERROR,
            category=ErrorCategory.ML_MODEL,
            http_status=500,
            internal_message=internal_message
        )


class EmbeddingException(IntelliQueryException):
    """Raised when embedding generation fails"""
    
    def __init__(self, message: str = "Failed to generate embeddings", internal_message: Optional[str] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.EMBEDDING_ERROR,
            category=ErrorCategory.EXTERNAL_SERVICE,
            http_status=500,
            internal_message=internal_message
        )


class LLMException(IntelliQueryException):
    """Raised when LLM operations fail"""
    
    def __init__(self, message: str = "Failed to generate response", internal_message: Optional[str] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.LLM_ERROR,
            category=ErrorCategory.EXTERNAL_SERVICE,
            http_status=500,
            internal_message=internal_message
        )


class TimeoutException(IntelliQueryException):
    """Raised when an operation times out"""
    
    def __init__(self, operation: str = "Operation", timeout_seconds: int = 60):
        super().__init__(
            message=f"{operation} timed out after {timeout_seconds} seconds",
            error_code=ErrorCode.TIMEOUT_ERROR,
            category=ErrorCategory.INTERNAL,
            http_status=504,
            details={"timeout_seconds": timeout_seconds}
        )


class RateLimitException(IntelliQueryException):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, limit: str = "Rate limit", retry_after: Optional[int] = None):
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        
        super().__init__(
            message=f"{limit} exceeded. Please try again later.",
            error_code=ErrorCode.RATE_LIMITED,
            category=ErrorCategory.SECURITY,
            http_status=429,
            details=details
        )


class ConfigurationException(IntelliQueryException):
    """Raised when configuration is invalid"""
    
    def __init__(self, message: str = "Service not properly configured"):
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            category=ErrorCategory.INTERNAL,
            http_status=500
        )


def get_request_id(request: Optional[Request] = None) -> str:
    """Get or generate request ID for tracking"""
    if request and hasattr(request.state, 'request_id'):
        return request.state.request_id
    return str(uuid.uuid4())


def handle_exception(func):
    """
    Decorator to handle exceptions in API endpoints.
    Converts exceptions to proper JSON responses with logging.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = kwargs.get('request') or (args[0] if args and isinstance(args[0], Request) else None)
        request_id = get_request_id(request)
        
        try:
            return await func(*args, **kwargs)
        
        except IntelliQueryException as e:
            # Log with internal details
            log_msg = f"[{request_id}] {e.error_code.value}: {e.message}"
            if e.internal_message:
                log_msg += f" | Internal: {e.internal_message}"
            logger.warning(log_msg)
            
            return JSONResponse(
                status_code=e.http_status,
                content=e.to_response_dict(request_id)
            )
        
        except Exception as e:
            # Log full stack trace for unexpected errors
            logger.error(
                f"[{request_id}] Unhandled exception: {str(e)}\n{traceback.format_exc()}"
            )
            
            # Return safe generic error
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": {
                        "code": ErrorCode.INTERNAL_ERROR.value,
                        "message": "An unexpected error occurred. Please try again.",
                        "request_id": request_id
                    }
                }
            )
    
    return wrapper


def safe_error_response(
    error_code: ErrorCode,
    message: str,
    request_id: str,
    http_status: int = 500,
    details: Optional[Dict] = None
) -> JSONResponse:
    """Create a safe error response"""
    return JSONResponse(
        status_code=http_status,
        content={
            "success": False,
            "error": {
                "code": error_code.value,
                "message": message,
                "request_id": request_id,
                "details": details or {}
            }
        }
    )
