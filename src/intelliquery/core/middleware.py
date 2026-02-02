"""
Middleware - Enterprise-Grade Request Processing
================================================
Request/Response middleware for security, logging, and rate limiting.

Features:
- Request ID injection
- Rate limiting (in-memory, Redis-ready interface)
- Audit logging middleware
- Request timing
"""

import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
import asyncio

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("audit")


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10  # Max requests in 1 second


@dataclass
class RateLimitState:
    """Track rate limit state for a client"""
    minute_count: int = 0
    hour_count: int = 0
    last_minute_reset: float = field(default_factory=time.time)
    last_hour_reset: float = field(default_factory=time.time)
    burst_timestamps: list = field(default_factory=list)


class InMemoryRateLimiter:
    """
    In-memory rate limiter (single-instance).
    
    For production with multiple instances, replace with Redis-backed implementation.
    Interface is designed to be swappable with Redis version.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._clients: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = asyncio.Lock()
    
    def _get_client_key(self, request: Request) -> str:
        """Get unique client identifier"""
        # Use X-Forwarded-For if behind proxy, otherwise use client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    async def is_rate_limited(self, request: Request) -> tuple[bool, Optional[int]]:
        """
        Check if request should be rate limited.
        
        Returns:
            (is_limited, retry_after_seconds)
        """
        client_key = self._get_client_key(request)
        now = time.time()
        
        async with self._lock:
            state = self._clients[client_key]
            
            # Reset counters if time window passed
            if now - state.last_minute_reset > 60:
                state.minute_count = 0
                state.last_minute_reset = now
            
            if now - state.last_hour_reset > 3600:
                state.hour_count = 0
                state.last_hour_reset = now
            
            # Clean old burst timestamps
            state.burst_timestamps = [ts for ts in state.burst_timestamps if now - ts < 1]
            
            # Check burst limit
            if len(state.burst_timestamps) >= self.config.burst_limit:
                return True, 1
            
            # Check minute limit
            if state.minute_count >= self.config.requests_per_minute:
                retry_after = int(60 - (now - state.last_minute_reset))
                return True, max(1, retry_after)
            
            # Check hour limit
            if state.hour_count >= self.config.requests_per_hour:
                retry_after = int(3600 - (now - state.last_hour_reset))
                return True, max(1, retry_after)
            
            # Not limited - increment counters
            state.minute_count += 1
            state.hour_count += 1
            state.burst_timestamps.append(now)
            
            return False, None
    
    def get_remaining(self, request: Request) -> Dict[str, int]:
        """Get remaining requests for client"""
        client_key = self._get_client_key(request)
        state = self._clients.get(client_key, RateLimitState())
        
        return {
            "minute_remaining": max(0, self.config.requests_per_minute - state.minute_count),
            "hour_remaining": max(0, self.config.requests_per_hour - state.hour_count)
        }


# Global rate limiter instance
rate_limiter = InMemoryRateLimiter()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request context (request ID, timing).
    Should be first middleware in chain.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Add response headers
        process_time = time.time() - request.state.start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}s"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.
    Applies rate limits to API endpoints.
    """
    
    def __init__(self, app: ASGIApp, limiter: Optional[InMemoryRateLimiter] = None):
        super().__init__(app)
        self.limiter = limiter or rate_limiter
        # Endpoints exempt from rate limiting
        self.exempt_paths = {"/health", "/metrics", "/docs", "/openapi.json", "/"}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)
        
        # Check rate limit
        is_limited, retry_after = await self.limiter.is_rate_limited(request)
        
        if is_limited:
            request_id = getattr(request.state, 'request_id', 'unknown')
            client_ip = self.limiter._get_client_key(request)
            
            logger.warning(f"[{request_id}] Rate limit exceeded for {client_ip}")
            
            return Response(
                content='{"success": false, "error": {"code": "RATE_LIMITED", "message": "Too many requests. Please slow down."}}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after)
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        remaining = self.limiter.get_remaining(request)
        response.headers["X-RateLimit-Remaining-Minute"] = str(remaining["minute_remaining"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(remaining["hour_remaining"])
        
        return response


class AuditLogMiddleware(BaseHTTPMiddleware):
    """
    Audit logging middleware.
    Logs all API requests for compliance and debugging.
    """
    
    def __init__(self, app: ASGIApp, log_request_body: bool = False):
        super().__init__(app)
        self.log_request_body = log_request_body
        # Paths to exclude from audit logging
        self.exclude_paths = {"/health", "/metrics", "/docs", "/openapi.json"}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        start_time = getattr(request.state, 'start_time', time.time())
        
        # Get client info
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.client.host if request.client else "unknown"
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Build audit log entry
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params) if request.query_params else None,
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", "unknown")[:100],
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            # User ID would be populated by auth middleware
            "user_id": getattr(request.state, 'user_id', None)
        }
        
        # Log at appropriate level based on status
        if response.status_code >= 500:
            audit_logger.error(f"AUDIT: {audit_entry}")
        elif response.status_code >= 400:
            audit_logger.warning(f"AUDIT: {audit_entry}")
        else:
            audit_logger.info(f"AUDIT: {audit_entry}")
        
        return response


def timeout_decorator(seconds: int = 60):
    """
    Decorator to add timeout to async functions.
    
    Usage:
        @timeout_decorator(30)
        async def slow_operation():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                from .error_handler import TimeoutException
                raise TimeoutException(operation=func.__name__, timeout_seconds=seconds)
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Simple circuit breaker for external service calls.
    
    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Too many failures, requests fail fast
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "CLOSED"
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> str:
        return self._state
    
    async def _check_state(self) -> str:
        """Check and potentially update circuit state"""
        async with self._lock:
            if self._state == "OPEN":
                if self._last_failure_time:
                    if time.time() - self._last_failure_time > self.recovery_timeout:
                        self._state = "HALF_OPEN"
                        self._half_open_calls = 0
                        logger.info("Circuit breaker entering HALF_OPEN state")
            return self._state
    
    async def call(self, func: Callable, *args, **kwargs):
        """
        Execute function through circuit breaker.
        
        Raises:
            Exception: If circuit is open
        """
        state = await self._check_state()
        
        if state == "OPEN":
            raise Exception("Circuit breaker is OPEN - service unavailable")
        
        if state == "HALF_OPEN":
            async with self._lock:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise Exception("Circuit breaker HALF_OPEN - max test calls reached")
                self._half_open_calls += 1
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Success - reset on HALF_OPEN
            async with self._lock:
                if self._state == "HALF_OPEN":
                    self._state = "CLOSED"
                    self._failure_count = 0
                    logger.info("Circuit breaker recovered - now CLOSED")
                elif self._state == "CLOSED":
                    self._failure_count = 0
            
            return result
            
        except Exception as e:
            await self._record_failure()
            raise
    
    async def _record_failure(self):
        """Record a failure and potentially open circuit"""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = "OPEN"
                logger.warning(f"Circuit breaker OPEN after {self._failure_count} failures")
            elif self._state == "HALF_OPEN":
                self._state = "OPEN"
                logger.warning("Circuit breaker back to OPEN after HALF_OPEN failure")


# Pre-configured circuit breakers for external services
llm_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
embedding_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=20)
database_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=10)
