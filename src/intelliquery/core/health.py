"""
Health Check Module - Comprehensive System Health Monitoring
=============================================================
Enterprise-grade health checks for all system components.

Features:
- Component-level health checks
- Degraded state detection
- Dependency health verification
- Prometheus-compatible metrics
"""

import time
import logging
import psutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component"""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SystemHealth:
    """Overall system health status"""
    status: HealthStatus
    version: str
    uptime_seconds: float
    components: Dict[str, ComponentHealth]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response dictionary"""
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "timestamp": self.timestamp.isoformat(),
            "components": {
                name: {
                    "status": comp.status.value,
                    "message": comp.message,
                    "latency_ms": comp.latency_ms,
                    "details": comp.details
                }
                for name, comp in self.components.items()
            }
        }


class HealthChecker:
    """
    Comprehensive health checker for all system components.
    """
    
    def __init__(self, app_version: str = "1.0.0"):
        self.app_version = app_version
        self._start_time = time.time()
        
        # Thresholds
        self.disk_warning_threshold = 80  # percent
        self.disk_critical_threshold = 95
        self.memory_warning_threshold = 80
        self.memory_critical_threshold = 95
        self.db_latency_warning_ms = 1000
        self.db_latency_critical_ms = 5000
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time
    
    async def check_database(self) -> ComponentHealth:
        """Check database connectivity and performance"""
        start = time.time()
        
        try:
            from ..core.database import db_client
            
            # Simple connectivity test
            result = db_client.test_connection()
            latency_ms = (time.time() - start) * 1000
            
            if not result:
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    message="Database connection test failed",
                    latency_ms=latency_ms
                )
            
            # Check latency
            if latency_ms > self.db_latency_critical_ms:
                status = HealthStatus.DEGRADED
                message = f"High latency: {latency_ms:.0f}ms"
            elif latency_ms > self.db_latency_warning_ms:
                status = HealthStatus.DEGRADED
                message = f"Elevated latency: {latency_ms:.0f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = "Connected"
            
            # Get pool stats if available
            pool_stats = {}
            if hasattr(db_client, 'get_pool_stats'):
                pool_stats = db_client.get_pool_stats()
            
            return ComponentHealth(
                name="database",
                status=status,
                message=message,
                latency_ms=round(latency_ms, 2),
                details={"pool": pool_stats}
            )
            
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Connection failed: {str(e)[:100]}",
                latency_ms=(time.time() - start) * 1000
            )
    
    async def check_embedding_service(self) -> ComponentHealth:
        """Check embedding endpoint availability"""
        start = time.time()
        
        try:
            from ..core.config import config
            
            if not config.EMBEDDING_ENDPOINT:
                return ComponentHealth(
                    name="embedding_service",
                    status=HealthStatus.DEGRADED,
                    message="Not configured - using mock embeddings"
                )
            
            # Try to get an embedding
            from ..core.database import db_client
            embedding = db_client.get_embedding("health check test")
            latency_ms = (time.time() - start) * 1000
            
            if embedding and len(embedding) > 0:
                return ComponentHealth(
                    name="embedding_service",
                    status=HealthStatus.HEALTHY,
                    message="Operational",
                    latency_ms=round(latency_ms, 2),
                    details={"embedding_dimensions": len(embedding)}
                )
            else:
                return ComponentHealth(
                    name="embedding_service",
                    status=HealthStatus.DEGRADED,
                    message="Empty embedding returned",
                    latency_ms=round(latency_ms, 2)
                )
                
        except Exception as e:
            return ComponentHealth(
                name="embedding_service",
                status=HealthStatus.UNHEALTHY,
                message=f"Service error: {str(e)[:100]}",
                latency_ms=(time.time() - start) * 1000
            )
    
    async def check_llm_service(self) -> ComponentHealth:
        """Check LLM endpoint availability"""
        try:
            from ..core.config import config
            
            if not config.LLM_ENDPOINT:
                return ComponentHealth(
                    name="llm_service",
                    status=HealthStatus.DEGRADED,
                    message="Not configured - using mock responses"
                )
            
            # Just verify endpoint is configured, don't make actual call
            return ComponentHealth(
                name="llm_service",
                status=HealthStatus.HEALTHY,
                message="Configured",
                details={"endpoint": config.LLM_ENDPOINT}
            )
            
        except Exception as e:
            return ComponentHealth(
                name="llm_service",
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {str(e)[:100]}"
            )
    
    def check_disk_space(self) -> ComponentHealth:
        """Check available disk space"""
        try:
            disk = psutil.disk_usage('/')
            percent_used = disk.percent
            free_gb = disk.free / (1024 ** 3)
            
            if percent_used >= self.disk_critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"Critical: {percent_used:.1f}% used"
            elif percent_used >= self.disk_warning_threshold:
                status = HealthStatus.DEGRADED
                message = f"Warning: {percent_used:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"{percent_used:.1f}% used"
            
            return ComponentHealth(
                name="disk_space",
                status=status,
                message=message,
                details={
                    "percent_used": round(percent_used, 1),
                    "free_gb": round(free_gb, 2),
                    "total_gb": round(disk.total / (1024 ** 3), 2)
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {str(e)[:100]}"
            )
    
    def check_memory(self) -> ComponentHealth:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            percent_used = memory.percent
            available_gb = memory.available / (1024 ** 3)
            
            if percent_used >= self.memory_critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"Critical: {percent_used:.1f}% used"
            elif percent_used >= self.memory_warning_threshold:
                status = HealthStatus.DEGRADED
                message = f"Warning: {percent_used:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"{percent_used:.1f}% used"
            
            return ComponentHealth(
                name="memory",
                status=status,
                message=message,
                details={
                    "percent_used": round(percent_used, 1),
                    "available_gb": round(available_gb, 2),
                    "total_gb": round(memory.total / (1024 ** 3), 2)
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {str(e)[:100]}"
            )
    
    def check_ml_model(self) -> ComponentHealth:
        """Check ML model status"""
        try:
            from ..ml.predictor import churn_predictor
            
            if not churn_predictor.is_trained:
                return ComponentHealth(
                    name="ml_model",
                    status=HealthStatus.DEGRADED,
                    message="Model not trained",
                    details={"trained": False}
                )
            
            stats = churn_predictor.training_stats or {}
            return ComponentHealth(
                name="ml_model",
                status=HealthStatus.HEALTHY,
                message="Trained and ready",
                details={
                    "trained": True,
                    "algorithm": stats.get("algorithm"),
                    "accuracy": stats.get("accuracy")
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="ml_model",
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {str(e)[:100]}"
            )
    
    async def get_health(self, deep_check: bool = False) -> SystemHealth:
        """
        Get comprehensive system health status.
        
        Args:
            deep_check: If True, performs slower checks like embedding service
            
        Returns:
            SystemHealth with all component statuses
        """
        components: Dict[str, ComponentHealth] = {}
        
        # Always run basic checks
        components["disk_space"] = self.check_disk_space()
        components["memory"] = self.check_memory()
        components["ml_model"] = self.check_ml_model()
        
        # Database check
        components["database"] = await self.check_database()
        
        # Deep checks (slower, optional)
        if deep_check:
            components["embedding_service"] = await self.check_embedding_service()
            components["llm_service"] = await self.check_llm_service()
        
        # Determine overall status
        statuses = [c.status for c in components.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        else:
            overall = HealthStatus.DEGRADED
        
        return SystemHealth(
            status=overall,
            version=self.app_version,
            uptime_seconds=self.uptime_seconds,
            components=components
        )
    
    def get_liveness(self) -> Dict[str, Any]:
        """
        Simple liveness check - is the process running?
        For Kubernetes liveness probes.
        """
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_readiness(self) -> Dict[str, Any]:
        """
        Readiness check - is the service ready to accept traffic?
        For Kubernetes readiness probes.
        """
        db_health = await self.check_database()
        
        is_ready = db_health.status != HealthStatus.UNHEALTHY
        
        return {
            "ready": is_ready,
            "status": "ready" if is_ready else "not_ready",
            "database": db_health.status.value,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global health checker instance
health_checker = HealthChecker(app_version="1.0.0")
