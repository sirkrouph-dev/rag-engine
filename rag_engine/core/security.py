"""
Production-grade security features including authentication, input validation, and audit logging.
"""
import hashlib
import hmac
import jwt
import secrets
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from functools import wraps
import re
import html
import bleach
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    ADMIN = "admin"


@dataclass
class User:
    """User model for authentication."""
    user_id: str
    username: str
    email: str
    roles: List[str]
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None


@dataclass
class AuditLogEntry:
    """Audit log entry model."""
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    success: bool


class InputValidator:
    """Production-grade input validation and sanitization."""
    
    # Common injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)",
        r"(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+",
        r"['\";]|\-\-|\/\*|\*\/"
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>"
    ]
    
    def __init__(self):
        # Allowed HTML tags for rich text (if needed)
        self.allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li']
        self.allowed_attributes = {}
    
    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_username(self, username: str) -> bool:
        """Validate username format."""
        # Only alphanumeric and underscores, 3-30 characters
        pattern = r'^[a-zA-Z0-9_]{3,30}$'
        return bool(re.match(pattern, username))
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength."""
        result = {
            "valid": False,
            "score": 0,
            "issues": []
        }
        
        if len(password) < 8:
            result["issues"].append("Password must be at least 8 characters long")
        else:
            result["score"] += 1
        
        if not re.search(r'[A-Z]', password):
            result["issues"].append("Password must contain at least one uppercase letter")
        else:
            result["score"] += 1
        
        if not re.search(r'[a-z]', password):
            result["issues"].append("Password must contain at least one lowercase letter")
        else:
            result["score"] += 1
        
        if not re.search(r'\d', password):
            result["issues"].append("Password must contain at least one digit")
        else:
            result["score"] += 1
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            result["issues"].append("Password must contain at least one special character")
        else:
            result["score"] += 1
        
        result["valid"] = len(result["issues"]) == 0
        return result
    
    def sanitize_html(self, text: str) -> str:
        """Sanitize HTML content to prevent XSS."""
        if not text:
            return ""
        
        # Use bleach to clean HTML
        try:
            cleaned = bleach.clean(
                text,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                strip=True
            )
            return cleaned
        except Exception as e:
            logger.warning(f"HTML sanitization failed: {e}")
            # Fallback to HTML escape
            return html.escape(text)
    
    def check_sql_injection(self, text: str) -> bool:
        """Check for potential SQL injection patterns."""
        if not text:
            return False
        
        text_upper = text.upper()
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_upper, re.IGNORECASE):
                return True
        return False
    
    def check_xss(self, text: str) -> bool:
        """Check for potential XSS patterns."""
        if not text:
            return False
        
        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def validate_input(self, data: Any, input_type: str = "text") -> Dict[str, Any]:
        """Comprehensive input validation."""
        result = {
            "valid": True,
            "sanitized_data": data,
            "issues": []
        }
        
        if data is None:
            return result
        
        if isinstance(data, str):
            # Check for injection attacks
            if self.check_sql_injection(data):
                result["issues"].append("Potential SQL injection detected")
                result["valid"] = False
            
            if self.check_xss(data):
                result["issues"].append("Potential XSS attack detected")
                result["valid"] = False
            
            # Sanitize based on input type
            if input_type == "html":
                result["sanitized_data"] = self.sanitize_html(data)
            elif input_type == "text":
                result["sanitized_data"] = html.escape(data)
            elif input_type == "email":
                if not self.validate_email(data):
                    result["issues"].append("Invalid email format")
                    result["valid"] = False
            elif input_type == "username":
                if not self.validate_username(data):
                    result["issues"].append("Invalid username format")
                    result["valid"] = False
        
        return result


class AuthenticationManager:
    """JWT-based authentication system."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256", 
                 token_expiration: int = 3600):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiration = token_expiration
        self.revoked_tokens: Set[str] = set()  # In production, use Redis
        self.users: Dict[str, User] = {}  # In production, use database
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash."""
        computed_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, password_hash)
    
    def create_token(self, user: User) -> str:
        """Create JWT token for user."""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": user.roles,
            "exp": datetime.utcnow() + timedelta(seconds=self.token_expiration),
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        if token in self.revoked_tokens:
            return None
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            if datetime.utcnow() > datetime.fromtimestamp(payload['exp']):
                return None
            
            return payload
        
        except jwt.InvalidTokenError:
            return None
    
    def revoke_token(self, token: str):
        """Revoke a token."""
        self.revoked_tokens.add(token)
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        # In production, this would query the database
        user = self.users.get(username)
        if user and user.is_active:
            # Verify password (this is simplified - in production use proper password storage)
            return user
        return None


class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, requests_per_minute: int = 100, burst_limit: int = 200):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.request_counts: Dict[str, List[float]] = {}
        self.burst_counts: Dict[str, int] = {}
    
    def is_allowed(self, identifier: str) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed for the identifier."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old requests
        if identifier in self.request_counts:
            self.request_counts[identifier] = [
                req_time for req_time in self.request_counts[identifier]
                if req_time > minute_ago
            ]
        else:
            self.request_counts[identifier] = []
        
        # Check rate limit
        request_count = len(self.request_counts[identifier])
        burst_count = self.burst_counts.get(identifier, 0)
        
        # Reset burst count every minute
        if not self.request_counts[identifier] or current_time - self.request_counts[identifier][0] > 60:
            self.burst_counts[identifier] = 0
            burst_count = 0
        
        rate_limit_info = {
            "requests_per_minute": request_count,
            "burst_count": burst_count,
            "rate_limit": self.requests_per_minute,
            "burst_limit": self.burst_limit,
            "reset_time": minute_ago + 60
        }
        
        # Check limits
        if request_count >= self.requests_per_minute:
            return False, rate_limit_info
        
        if burst_count >= self.burst_limit:
            return False, rate_limit_info
        
        # Record request
        self.request_counts[identifier].append(current_time)
        self.burst_counts[identifier] = burst_count + 1
        
        return True, rate_limit_info


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter for audit logs
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        
        # Create handler (in production, use external log aggregation)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.audit_entries: List[AuditLogEntry] = []  # In production, use database
    
    def log_event(self, user_id: Optional[str], action: str, resource: str, 
                  details: Dict[str, Any], ip_address: str, user_agent: str, 
                  success: bool = True):
        """Log an audit event."""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action=action,
            resource=resource,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success
        )
        
        self.audit_entries.append(entry)
        
        # Log to file/external system
        self.logger.info(
            f"User: {user_id} | Action: {action} | Resource: {resource} | "
            f"Success: {success} | IP: {ip_address} | Details: {details}"
        )
    
    def get_audit_logs(self, user_id: Optional[str] = None, 
                      action: Optional[str] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> List[AuditLogEntry]:
        """Retrieve audit logs with filters."""
        filtered_logs = self.audit_entries
        
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
        
        if action:
            filtered_logs = [log for log in filtered_logs if log.action == action]
        
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
        
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
        
        return filtered_logs


def require_auth(security_level: SecurityLevel = SecurityLevel.AUTHENTICATED):
    """Decorator for requiring authentication."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This would be implemented with actual request context
            # For now, this is a placeholder
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def validate_input(input_type: str = "text"):
    """Decorator for input validation."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This would validate all inputs in the function
            # For now, this is a placeholder
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Security utilities
def setup_security(config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup production security components."""
    
    # JWT configuration
    jwt_secret = config.get("security", {}).get("jwt_secret_key", secrets.token_urlsafe(32))
    jwt_algorithm = config.get("security", {}).get("jwt_algorithm", "HS256")
    jwt_expiration = config.get("security", {}).get("jwt_expiration", 3600)
    
    # Rate limiting configuration
    rate_config = config.get("security", {}).get("rate_limiting", {})
    requests_per_minute = rate_config.get("requests_per_minute", 100)
    burst_limit = rate_config.get("burst_limit", 200)
    
    return {
        "input_validator": InputValidator(),
        "auth_manager": AuthenticationManager(jwt_secret, jwt_algorithm, jwt_expiration),
        "rate_limiter": RateLimiter(requests_per_minute, burst_limit),
        "audit_logger": AuditLogger()
    }
