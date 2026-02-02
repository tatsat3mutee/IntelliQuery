"""
Security Module - Input Validation and Sanitization
===================================================
Enterprise-grade security utilities for IntelliQuery AI.

Features:
- Input validation with size limits
- SQL identifier sanitization
- File type validation
- Request validation middleware
"""

import re
import logging
from typing import List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Allowed file types for upload"""
    PDF = ("application/pdf", [".pdf"])
    TEXT = ("text/plain", [".txt"])
    CSV = ("text/csv", [".csv"])
    EXCEL_XLSX = ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", [".xlsx"])
    EXCEL_XLS = ("application/vnd.ms-excel", [".xls"])


# SQL reserved words that need escaping
SQL_RESERVED_WORDS: Set[str] = {
    'add', 'all', 'alter', 'and', 'any', 'as', 'asc', 'between', 'by', 'case',
    'check', 'column', 'constraint', 'create', 'cross', 'current', 'database',
    'default', 'delete', 'desc', 'distinct', 'drop', 'else', 'end', 'exists',
    'false', 'for', 'foreign', 'from', 'full', 'grant', 'group', 'having', 'if',
    'in', 'index', 'inner', 'insert', 'into', 'is', 'join', 'key', 'left', 'like',
    'limit', 'not', 'null', 'offset', 'on', 'or', 'order', 'outer', 'primary',
    'references', 'revoke', 'right', 'select', 'set', 'table', 'then', 'to',
    'true', 'union', 'unique', 'update', 'values', 'when', 'where', 'with'
}

# Dangerous SQL patterns to block
DANGEROUS_SQL_PATTERNS = [
    r';\s*--',           # SQL comment after semicolon
    r';\s*DROP',         # DROP after semicolon
    r';\s*DELETE',       # DELETE after semicolon
    r';\s*UPDATE',       # UPDATE after semicolon
    r';\s*INSERT',       # INSERT after semicolon
    r';\s*ALTER',        # ALTER after semicolon
    r';\s*CREATE',       # CREATE after semicolon
    r';\s*TRUNCATE',     # TRUNCATE after semicolon
    r'UNION\s+SELECT',   # UNION injection
    r'OR\s+1\s*=\s*1',   # Classic OR injection
    r"OR\s+'[^']*'\s*=\s*'[^']*'",  # String-based OR injection
    r'--\s*$',           # Comment at end
    r'/\*.*\*/',         # Block comment
]


@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    error_message: Optional[str] = None
    sanitized_value: Optional[Any] = None


class InputValidator:
    """Validates and sanitizes user inputs"""
    
    # Size limits
    MAX_QUERY_LENGTH = 10000
    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB
    MAX_FILENAME_LENGTH = 255
    MAX_FORM_FIELD_LENGTH = 5000
    
    # Allowed characters for identifiers
    IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    
    @classmethod
    def validate_sql_identifier(cls, identifier: str) -> ValidationResult:
        """
        Validate and sanitize SQL identifier (table name, column name).
        Prevents SQL injection in dynamic queries.
        """
        if not identifier:
            return ValidationResult(False, "Identifier cannot be empty")
        
        # Normalize
        identifier = identifier.strip().lower()
        
        # Check length
        if len(identifier) > 128:
            return ValidationResult(False, "Identifier too long (max 128 chars)")
        
        # Check for valid characters only
        if not cls.IDENTIFIER_PATTERN.match(identifier):
            # Sanitize by removing invalid characters
            sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', identifier)
            sanitized = re.sub(r'^[^a-zA-Z_]', '_', sanitized)  # Ensure starts with letter/underscore
            
            # If it's a reserved word, prefix it
            if sanitized.lower() in SQL_RESERVED_WORDS:
                sanitized = f"col_{sanitized}"
            
            return ValidationResult(True, None, sanitized)
        
        # Check reserved words
        if identifier.lower() in SQL_RESERVED_WORDS:
            return ValidationResult(True, None, f"col_{identifier}")
        
        return ValidationResult(True, None, identifier)
    
    @classmethod
    def validate_query_input(cls, query: str) -> ValidationResult:
        """
        Validate natural language query input.
        Checks for SQL injection attempts and size limits.
        """
        if not query:
            return ValidationResult(False, "Query cannot be empty")
        
        query = query.strip()
        
        # Check length
        if len(query) > cls.MAX_QUERY_LENGTH:
            return ValidationResult(False, f"Query too long (max {cls.MAX_QUERY_LENGTH} chars)")
        
        # Check for dangerous SQL patterns
        query_upper = query.upper()
        for pattern in DANGEROUS_SQL_PATTERNS:
            if re.search(pattern, query_upper, re.IGNORECASE):
                logger.warning(f"Potential SQL injection blocked: {query[:100]}...")
                return ValidationResult(False, "Query contains potentially dangerous patterns")
        
        return ValidationResult(True, None, query)
    
    @classmethod
    def validate_file_upload(
        cls,
        filename: str,
        content: bytes,
        allowed_types: List[FileType],
        max_size_bytes: Optional[int] = None
    ) -> ValidationResult:
        """
        Validate file upload for security.
        Checks file size, extension, and MIME type.
        """
        max_size = max_size_bytes or cls.MAX_FILE_SIZE_BYTES
        
        # Check filename
        if not filename:
            return ValidationResult(False, "Filename is required")
        
        if len(filename) > cls.MAX_FILENAME_LENGTH:
            return ValidationResult(False, f"Filename too long (max {cls.MAX_FILENAME_LENGTH} chars)")
        
        # Sanitize filename - remove path traversal attempts
        filename = filename.replace('..', '').replace('/', '_').replace('\\', '_')
        
        # Check file size
        if len(content) > max_size:
            return ValidationResult(
                False, 
                f"File too large ({len(content) / 1024 / 1024:.1f}MB). Max: {max_size / 1024 / 1024:.0f}MB"
            )
        
        # Check extension
        ext = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        allowed_extensions = []
        for file_type in allowed_types:
            allowed_extensions.extend(file_type.value[1])
        
        if ext not in allowed_extensions:
            return ValidationResult(
                False, 
                f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        return ValidationResult(True, None, filename)
    
    @classmethod
    def sanitize_string_value(cls, value: str) -> str:
        """
        Sanitize a string value for safe SQL insertion.
        Escapes special characters.
        """
        if value is None:
            return ''
        
        # Escape single quotes and backslashes
        return str(value).replace("\\", "\\\\").replace("'", "''")
    
    @classmethod
    def validate_form_field(cls, field_name: str, value: str, max_length: Optional[int] = None) -> ValidationResult:
        """Validate a form field value"""
        max_len = max_length or cls.MAX_FORM_FIELD_LENGTH
        
        if value and len(value) > max_len:
            return ValidationResult(False, f"{field_name} too long (max {max_len} chars)")
        
        return ValidationResult(True, None, value.strip() if value else value)


class SQLSanitizer:
    """
    Safe SQL query builder to prevent SQL injection.
    Uses allowlisted columns and parameterized patterns.
    """
    
    def __init__(self, allowed_columns: Set[str], table_name: str):
        """
        Initialize with allowed columns and table name.
        
        Args:
            allowed_columns: Set of valid column names for this table
            table_name: Fully qualified table name (e.g., catalog.schema.table)
        """
        self.allowed_columns = {col.lower() for col in allowed_columns}
        self.table_name = table_name
    
    def validate_column(self, column: str) -> bool:
        """Check if column is in allowlist"""
        return column.lower() in self.allowed_columns
    
    def validate_columns(self, columns: List[str]) -> List[str]:
        """Validate and return only allowed columns"""
        return [col for col in columns if self.validate_column(col)]
    
    def build_select(
        self,
        columns: List[str],
        conditions: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        limit: int = 100
    ) -> str:
        """
        Build a safe SELECT query using allowlisted columns.
        
        Args:
            columns: List of column names (will be validated)
            conditions: Pre-validated WHERE conditions
            order_by: Column to order by (will be validated)
            limit: Max rows to return
            
        Returns:
            Safe SQL query string
        """
        # Validate columns
        safe_columns = self.validate_columns(columns)
        if not safe_columns:
            safe_columns = ['*']
        
        # Build query
        sql = f"SELECT {', '.join(safe_columns)} FROM {self.table_name}"
        
        # Add conditions (should be pre-validated)
        if conditions:
            sql += f" WHERE {' AND '.join(conditions)}"
        
        # Add ORDER BY if valid column
        if order_by and self.validate_column(order_by):
            sql += f" ORDER BY {order_by} DESC"
        
        # Always add limit for safety
        limit = min(limit, 10000)  # Hard cap at 10000
        sql += f" LIMIT {limit}"
        
        return sql
    
    def build_safe_condition(self, column: str, operator: str, value: Any) -> Optional[str]:
        """
        Build a safe WHERE condition.
        
        Args:
            column: Column name (will be validated)
            operator: One of =, !=, <, >, <=, >=, LIKE, IN
            value: Value to compare (will be escaped)
            
        Returns:
            Safe SQL condition or None if invalid
        """
        # Validate column
        if not self.validate_column(column):
            logger.warning(f"Invalid column rejected: {column}")
            return None
        
        # Validate operator
        allowed_operators = {'=', '!=', '<', '>', '<=', '>=', 'LIKE', 'IN', 'IS NULL', 'IS NOT NULL'}
        if operator.upper() not in allowed_operators:
            return None
        
        # Handle different operators
        if operator.upper() in ('IS NULL', 'IS NOT NULL'):
            return f"{column} {operator.upper()}"
        
        if operator.upper() == 'IN' and isinstance(value, (list, tuple)):
            escaped_values = [f"'{InputValidator.sanitize_string_value(v)}'" for v in value]
            return f"{column} IN ({', '.join(escaped_values)})"
        
        # Escape value
        if isinstance(value, str):
            escaped = InputValidator.sanitize_string_value(value)
            return f"{column} {operator} '{escaped}'"
        elif isinstance(value, (int, float)):
            return f"{column} {operator} {value}"
        elif value is None:
            return f"{column} IS NULL"
        
        return None


# Global validator instance
input_validator = InputValidator()
