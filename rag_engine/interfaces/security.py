"""
Authentication and security framework for API customization.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
import jwt
import hashlib
import time
from dataclasses import dataclass


@dataclass
class SecurityConfig:
    """Security configuration for APIs."""
    
    # Authentication
    enable_auth: bool = False
    auth_method: str = "api_key"  # api_key, jwt, oauth2, basic
    api_keys: List[str] = None
    jwt_secret: str = None
    jwt_algorithm: str = "HS256"
    jwt_expiry: int = 3600  # seconds
    
    # Rate Limiting
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10
    rate_limit_storage: str = "memory"  # memory, redis
    
    # CORS
    cors_origins: List[str] = None
    cors_methods: List[str] = None
    cors_headers: List[str] = None
    
    # Security Headers
    enable_security_headers: bool = True
    security_headers: Dict[str, str] = None
    
    # IP Filtering
    allowed_ips: List[str] = None
    blocked_ips: List[str] = None
    
    def __post_init__(self):
        if self.api_keys is None:
            self.api_keys = []
        if self.cors_origins is None:
            self.cors_origins = ["*"]
        if self.cors_methods is None:
            self.cors_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        if self.cors_headers is None:
            self.cors_headers = ["*"]
        if self.security_headers is None:
            self.security_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
            }


class AuthenticationProvider(ABC):
    """Base authentication provider."""
    
    @abstractmethod
    def authenticate(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate request and return user info or None."""
        pass
    
    @abstractmethod
    def generate_token(self, user_data: Dict[str, Any]) -> str:
        """Generate authentication token."""
        pass


class APIKeyProvider(AuthenticationProvider):
    """API Key authentication provider."""
    
    def __init__(self, valid_keys: List[str]):
        self.valid_keys = set(valid_keys)
    
    def authenticate(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate via API key."""
        api_key = (
            request_data.get("headers", {}).get("X-API-Key") or
            request_data.get("headers", {}).get("Authorization", "").replace("Bearer ", "") or
            request_data.get("query_params", {}).get("api_key")
        )
        
        if api_key in self.valid_keys:
            return {"user_id": hashlib.sha256(api_key.encode()).hexdigest()[:8], "api_key": api_key}
        return None
    
    def generate_token(self, user_data: Dict[str, Any]) -> str:
        """Generate API key."""
        return hashlib.sha256(f"{user_data}_{time.time()}".encode()).hexdigest()


class JWTProvider(AuthenticationProvider):
    """JWT authentication provider."""
    
    def __init__(self, secret: str, algorithm: str = "HS256", expiry: int = 3600):
        self.secret = secret
        self.algorithm = algorithm
        self.expiry = expiry
    
    def authenticate(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate via JWT token."""
        auth_header = request_data.get("headers", {}).get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.replace("Bearer ", "")
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        except jwt.InvalidTokenError:
            return None
    
    def generate_token(self, user_data: Dict[str, Any]) -> str:
        """Generate JWT token."""
        payload = {
            **user_data,
            "exp": time.time() + self.expiry,
            "iat": time.time()
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)


class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.requests = {}  # In-memory storage
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        minute_window = int(now // 60)
        
        if identifier not in self.requests:
            self.requests[identifier] = {}
        
        user_requests = self.requests[identifier]
        
        # Clean old windows
        old_windows = [w for w in user_requests.keys() if w < minute_window - 1]
        for w in old_windows:
            del user_requests[w]
        
        # Count requests in current window
        current_count = user_requests.get(minute_window, 0)
        
        if current_count >= self.requests_per_minute:
            return False
        
        # Allow request
        user_requests[minute_window] = current_count + 1
        return True
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        now = time.time()
        minute_window = int(now // 60)
        current_count = self.requests.get(identifier, {}).get(minute_window, 0)
        return max(0, self.requests_per_minute - current_count)


class SecurityManager:
    """Central security manager for API frameworks."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.auth_provider = self._create_auth_provider()
        self.rate_limiter = RateLimiter(
            config.rate_limit_per_minute,
            config.rate_limit_burst
        ) if config.enable_rate_limiting else None
    
    def _create_auth_provider(self) -> Optional[AuthenticationProvider]:
        """Create authentication provider based on config."""
        if not self.config.enable_auth:
            return None
        
        if self.config.auth_method == "api_key":
            return APIKeyProvider(self.config.api_keys)
        elif self.config.auth_method == "jwt":
            return JWTProvider(
                self.config.jwt_secret,
                self.config.jwt_algorithm,
                self.config.jwt_expiry
            )
        elif self.config.auth_method in ["none", "NONE"]:
            return None
        else:
            raise ValueError(f"Unsupported auth method: {self.config.auth_method}")
    
    def authenticate_request(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate incoming request."""
        if not self.auth_provider:
            return {"user_id": "anonymous"}
        
        return self.auth_provider.authenticate(request_data)
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check rate limit for identifier."""
        if not self.rate_limiter:
            return True
        
        return self.rate_limiter.is_allowed(identifier)
    
    def is_ip_allowed(self, ip: str) -> bool:
        """Check if IP is allowed."""
        if self.config.blocked_ips and ip in self.config.blocked_ips:
            return False
        
        if self.config.allowed_ips:
            # Check exact matches first
            if ip in self.config.allowed_ips:
                return True
            
            # Check CIDR notation matches
            import ipaddress
            try:
                ip_obj = ipaddress.ip_address(ip)
                for allowed_ip in self.config.allowed_ips:
                    try:
                        if '/' in allowed_ip:
                            # CIDR notation
                            network = ipaddress.ip_network(allowed_ip, strict=False)
                            if ip_obj in network:
                                return True
                        else:
                            # Single IP
                            if ip == allowed_ip:
                                return True
                    except ValueError:
                        # Invalid IP format, skip
                        continue
            except ValueError:
                # Invalid IP format
                return False
            
            # If we have allowed_ips but IP doesn't match any, deny
            return False
        
        return True
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers to add to responses."""
        if not self.config.enable_security_headers:
            return {}
        
        return self.config.security_headers.copy()
    
    def create_middleware(self, framework: str) -> Callable:
        """Create framework-specific security middleware."""
        
        def fastapi_middleware(request, call_next):
            """FastAPI security middleware."""
            # Extract request data
            request_data = {
                "headers": dict(request.headers),
                "query_params": dict(request.query_params),
                "client_ip": request.client.host
            }
            
            # IP filtering
            if not self.is_ip_allowed(request_data["client_ip"]):
                from fastapi import HTTPException
                raise HTTPException(status_code=403, detail="IP not allowed")
            
            # Rate limiting
            if not self.check_rate_limit(request_data["client_ip"]):
                from fastapi import HTTPException
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Authentication
            if self.config.enable_auth:
                user = self.authenticate_request(request_data)
                if not user:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=401, detail="Authentication required")
                request.state.user = user
            
            # Process request
            response = call_next(request)
            
            # Add security headers
            for header, value in self.get_security_headers().items():
                response.headers[header] = value
            
            return response
        
        def flask_middleware(app):
            """Flask security middleware."""
            from flask import request, jsonify, g
            
            @app.before_request
            def security_check():
                # Extract request data
                request_data = {
                    "headers": dict(request.headers),
                    "query_params": dict(request.args),
                    "client_ip": request.remote_addr
                }
                
                # IP filtering
                if not self.is_ip_allowed(request_data["client_ip"]):
                    return jsonify({"error": "IP not allowed"}), 403
                
                # Rate limiting
                if not self.check_rate_limit(request_data["client_ip"]):
                    return jsonify({"error": "Rate limit exceeded"}), 429
                
                # Authentication
                if self.config.enable_auth:
                    user = self.authenticate_request(request_data)
                    if not user:
                        return jsonify({"error": "Authentication required"}), 401
                    g.user = user
            
            @app.after_request
            def add_security_headers(response):
                for header, value in self.get_security_headers().items():
                    response.headers[header] = value
                return response
        
        if framework == "fastapi":
            return fastapi_middleware
        elif framework == "flask":
            return flask_middleware
        else:
            raise ValueError(f"Unsupported framework: {framework}")


def create_auth_decorator(security_manager: SecurityManager):
    """Create authentication decorator."""
    
    def require_auth(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would be implemented per framework
            return func(*args, **kwargs)
        return wrapper
    
    return require_auth
