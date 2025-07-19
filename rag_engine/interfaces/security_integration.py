"""
Security integration module for RAG Engine APIs.
Demonstrates how to properly integrate the comprehensive security framework.
"""
from typing import Dict, Any, Optional, List, Callable
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, APIKeyHeader, HTTPAuthorizationCredentials
from functools import wraps
import logging
import time

from .security import SecurityManager, SecurityConfig
from ..core.security import InputValidator, AuditLogger, AuthenticationManager
from ..core.reliability import CircuitBreaker, RetryHandler, CircuitBreakerConfig, RetryConfig

logger = logging.getLogger(__name__)


class SecurityIntegration:
    """Comprehensive security integration for API frameworks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize security components
        self.security_config = SecurityConfig(
            enable_auth=config.get("enable_auth", False),
            auth_method=config.get("auth_method", "none"),
            api_keys=config.get("api_keys", []),
            jwt_secret=config.get("jwt_secret", ""),
            enable_rate_limiting=config.get("enable_rate_limiting", True),
            rate_limit_per_minute=config.get("rate_limit_per_minute", 60),
            cors_origins=config.get("cors_origins", ["*"]),
            allowed_ips=config.get("allowed_ips", []),
            blocked_ips=config.get("blocked_ips", [])
        )
        
        self.security_manager = SecurityManager(self.security_config)
        self.input_validator = InputValidator()
        self.audit_logger = AuditLogger()
        
        # Initialize authentication manager
        if self.security_config.enable_auth and self.security_config.jwt_secret and self.security_config.jwt_secret.strip():
            self.auth_manager = AuthenticationManager(
                secret_key=self.security_config.jwt_secret,
                token_expiration=config.get("token_expiration", 3600)
            )
        else:
            self.auth_manager = None
            
        # Initialize reliability components
        circuit_config = CircuitBreakerConfig(
            failure_threshold=config.get("circuit_breaker_threshold", 5),
            recovery_timeout=config.get("circuit_breaker_timeout", 60)
        )
        self.circuit_breaker = CircuitBreaker(circuit_config)
        
        retry_config = RetryConfig(
            max_attempts=config.get("retry_max_attempts", 3),
            backoff_factor=config.get("retry_backoff", 2.0)
        )
        self.retry_handler = RetryHandler(retry_config)
        
        # Simple session storage for testing
        self._sessions = {}
    
    def create_fastapi_middleware(self) -> Callable:
        """Create FastAPI middleware with comprehensive security."""
        
        async def security_middleware(request: Request, call_next):
            # Record request start time
            start_time = time.time()
            
            # Extract request information
            client_ip = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "unknown")
            
            try:
                # 1. IP filtering
                if not self.security_manager.is_ip_allowed(client_ip):
                    self.audit_logger.log_event(
                        user_id=None,
                        action="access_denied",
                        resource=str(request.url.path),
                        details={"reason": "IP blocked"},
                        ip_address=client_ip,
                        user_agent=user_agent,
                        success=False
                    )
                    raise HTTPException(status_code=403, detail="Access denied")
                
                # 2. Rate limiting
                if not self.security_manager.check_rate_limit(client_ip):
                    self.audit_logger.log_event(
                        user_id=None,
                        action="rate_limit_exceeded",
                        resource=str(request.url.path),
                        details={"client_ip": client_ip},
                        ip_address=client_ip,
                        user_agent=user_agent,
                        success=False
                    )
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                # 3. Input validation for POST/PUT requests
                if request.method in ["POST", "PUT", "PATCH"]:
                    # This would validate request body in real implementation
                    # For now, we'll validate query parameters
                    for key, value in request.query_params.items():
                        validation_result = self.input_validator.validate_input(value, "text")
                        if not validation_result["valid"]:
                            self.audit_logger.log_event(
                                user_id=None,
                                action="invalid_input",
                                resource=str(request.url.path),
                                details={"issues": validation_result["issues"], "parameter": key},
                                ip_address=client_ip,
                                user_agent=user_agent,
                                success=False
                            )
                            raise HTTPException(status_code=400, detail="Invalid input detected")
                
                # 4. Process request with circuit breaker
                response = await self.circuit_breaker.call(call_next, request)
                
                # 5. Add security headers
                security_headers = self.security_manager.get_security_headers()
                if security_headers:
                    for header, value in security_headers.items():
                        response.headers[header] = value
                
                # 6. Log successful request
                processing_time = time.time() - start_time
                self.audit_logger.log_event(
                    user_id=getattr(request.state, 'user_id', None),
                    action="api_request",
                    resource=str(request.url.path),
                    details={
                        "method": request.method,
                        "status_code": response.status_code,
                        "processing_time": processing_time
                    },
                    ip_address=client_ip,
                    user_agent=user_agent,
                    success=True
                )
                
                return response
                
            except HTTPException:
                # Re-raise HTTP exceptions
                raise
            except Exception as e:
                # Log unexpected errors
                self.audit_logger.log_event(
                    user_id=getattr(request.state, 'user_id', None),
                    action="api_error",
                    resource=str(request.url.path),
                    details={"error": str(e)},
                    ip_address=client_ip,
                    user_agent=user_agent,
                    success=False
                )
                raise HTTPException(status_code=500, detail="Internal server error")
        
        return security_middleware
    
    def create_auth_dependency(self):
        """Create authentication dependency for FastAPI endpoints."""
        if not self.security_config.enable_auth:
            return None
            
        if self.security_config.auth_method == "api_key":
            api_key_header = APIKeyHeader(name="X-API-Key")
            
            async def verify_api_key(api_key: str = Depends(api_key_header)):
                if api_key not in self.security_config.api_keys:
                    raise HTTPException(status_code=401, detail="Invalid API key")
                return {"user_id": f"api_key_{api_key[:8]}", "auth_method": "api_key"}
            
            return verify_api_key
            
        elif self.security_config.auth_method == "jwt":
            bearer_scheme = HTTPBearer()
            
            async def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
                if not self.auth_manager:
                    raise HTTPException(status_code=500, detail="Authentication not configured")
                
                user_data = self.auth_manager.verify_token(credentials.credentials)
                if not user_data:
                    raise HTTPException(status_code=401, detail="Invalid or expired token")
                
                return user_data
            
            return verify_jwt
        
        return None
    
    def create_input_validation_decorator(self):
        """Create decorator for input validation."""
        
        def validate_inputs(input_types: Dict[str, str] = None):
            """Decorator to validate function inputs."""
            if input_types is None:
                input_types = {}
                
            def decorator(func):
                @wraps(func)
                async def wrapper(*args, **kwargs):
                    # Validate kwargs based on input_types
                    for param_name, input_type in input_types.items():
                        if param_name in kwargs:
                            value = kwargs[param_name]
                            if isinstance(value, str):
                                validation_result = self.input_validator.validate_input(value, input_type)
                                if not validation_result["valid"]:
                                    raise HTTPException(
                                        status_code=400,
                                        detail=f"Invalid {param_name}: {', '.join(validation_result['issues'])}"
                                    )
                                # Use sanitized data
                                kwargs[param_name] = validation_result["sanitized_data"]
                    
                    return await func(*args, **kwargs)
                return wrapper
            return decorator
        
        return validate_inputs
    
    def create_circuit_breaker_decorator(self, operation_name: str):
        """Create circuit breaker decorator for specific operations."""
        
        def circuit_breaker_protection(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await self.circuit_breaker.call(func, *args, **kwargs)
                except Exception as e:
                    logger.error(f"Circuit breaker triggered for {operation_name}: {e}")
                    raise HTTPException(
                        status_code=503,
                        detail=f"Service temporarily unavailable for {operation_name}"
                    )
            return wrapper
        
        return circuit_breaker_protection
    
    def create_retry_decorator(self, operation_name: str):
        """Create retry decorator for specific operations."""
        
        def retry_protection(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await self.retry_handler(func)(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Retry exhausted for {operation_name}: {e}")
                    raise HTTPException(
                        status_code=503,
                        detail=f"Operation failed after retries: {operation_name}"
                    )
            return wrapper
        
        return retry_protection

    # Additional methods for test compatibility
    def validate_input(self, data: str, input_type: str = "text") -> Dict[str, Any]:
        """Validate input data."""
        return self.input_validator.validate_input(data, input_type)
    
    def create_jwt_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT token for user data."""
        if not self.auth_manager:
            raise ValueError("Authentication manager not configured")
        # Convert dict to User object
        from ..core.security import User
        user = User(
            user_id=user_data.get("user_id", ""),
            username=user_data.get("username", ""),
            email=user_data.get("email", ""),
            roles=user_data.get("roles", [])
        )
        return self.auth_manager.create_token(user)
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        if not self.auth_manager:
            return None
        return self.auth_manager.verify_token(token)
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key."""
        return api_key in self.security_config.api_keys
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check rate limit for client."""
        return self.security_manager.check_rate_limit(client_id)
    
    def is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP address is allowed."""
        return self.security_manager.is_ip_allowed(ip_address)
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers."""
        headers = self.security_manager.get_security_headers()
        return headers if headers else {}
    
    def hash_password(self, password: str) -> str:
        """Hash password."""
        if not self.auth_manager:
            raise ValueError("Authentication manager not configured")
        password_hash, salt = self.auth_manager.hash_password(password)
        return f"{password_hash}:{salt}"
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        if not self.auth_manager:
            return False
        try:
            password_hash, salt = hashed.split(":", 1)
            return self.auth_manager.verify_password(password, password_hash, salt)
        except (ValueError, AttributeError):
            return False
    
    def create_session(self, user_id: str, session_data: Dict[str, Any]) -> str:
        """Create session for user."""
        # Simple session creation - in production, use proper session management
        import secrets
        session_id = secrets.token_urlsafe(32)
        # Store session data (in production, use database)
        self._sessions[session_id] = session_data
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        # Simple session retrieval - in production, use proper session management
        return self._sessions.get(session_id)
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session."""
        # Simple session invalidation - in production, use proper session management
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def generate_csrf_token(self) -> str:
        """Generate CSRF token."""
        import secrets
        return secrets.token_urlsafe(32)
    
    def validate_csrf_token(self, token: str, stored_token: str) -> bool:
        """Validate CSRF token."""
        return token == stored_token and len(token) > 0

    def log_audit_event(self, user_id: Optional[str], action: str, resource: str, 
                       details: Dict[str, Any], ip_address: str, user_agent: str, 
                       success: bool = True):
        """Log audit event."""
        self.audit_logger.log_event(
            user_id=user_id,
            action=action,
            resource=resource,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success
        )
    
    def sanitize_html(self, html_input: str) -> str:
        """Sanitize HTML input."""
        # First check for XSS patterns
        if self.input_validator.check_xss(html_input):
            # If XSS detected, strip all HTML and escape
            import html
            import re
            # Remove script tags and their content
            cleaned = re.sub(r'<script[^>]*>.*?</script>', '', html_input, flags=re.IGNORECASE | re.DOTALL)
            # Remove other dangerous patterns
            cleaned = re.sub(r'javascript:', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'on\w+\s*=', '', cleaned, flags=re.IGNORECASE)
            # Escape remaining HTML
            return html.escape(cleaned)
        
        # Otherwise use the standard sanitization
        return self.input_validator.sanitize_html(html_input)
    
    def get_fastapi_middleware(self):
        """Get FastAPI middleware."""
        return self.create_fastapi_middleware()


def apply_security_to_fastapi(app: FastAPI, config: Dict[str, Any]) -> SecurityIntegration:
    """Apply comprehensive security to a FastAPI application."""
    
    # Initialize security integration
    security = SecurityIntegration(config)
    
    # Add security middleware
    app.middleware("http")(security.create_fastapi_middleware())
    
    # Return security integration for use in endpoints
    return security


# Example usage
def create_secure_endpoint_example(security: SecurityIntegration):
    """Example of how to create a secure endpoint with all protections."""
    
    # Get authentication dependency
    auth_dependency = security.create_auth_dependency()
    dependencies = [Depends(auth_dependency)] if auth_dependency else []
    
    # Get validation decorator
    validate_inputs = security.create_input_validation_decorator()
    
    # Get reliability decorators
    circuit_breaker = security.create_circuit_breaker_decorator("chat")
    retry_protection = security.create_retry_decorator("chat")
    
    @validate_inputs({"query": "text", "session_id": "text"})
    @circuit_breaker
    @retry_protection
    async def secure_chat_endpoint(
        query: str,
        session_id: str = "default",
                 user: Optional[Dict[str, Any]] = Depends(auth_dependency) if auth_dependency else None
    ):
        """Fully secured chat endpoint with all protections applied."""
        
        # Your actual endpoint logic here
        return {
            "query": query,
            "session_id": session_id,
            "user": user,
            "response": "This is a secure response",
            "security_applied": [
                "authentication",
                "input_validation", 
                "circuit_breaker",
                "retry_logic",
                "rate_limiting",
                "audit_logging"
            ]
        }
    
    return secure_chat_endpoint 