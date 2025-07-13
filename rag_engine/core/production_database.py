"""
Production database integration for RAG Engine.
Handles user management, session storage, audit logs, and configuration persistence.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import asyncio
import hashlib
import secrets

logger = logging.getLogger(__name__)


@dataclass
class User:
    """User model for database storage."""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    roles: List[str]
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Session:
    """Session model for database storage."""
    session_id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


@dataclass
class AuditLog:
    """Audit log model for database storage."""
    log_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    timestamp: datetime
    success: bool
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DatabaseProvider:
    """Base class for database providers."""
    
    async def connect(self):
        """Connect to the database."""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from the database."""
        raise NotImplementedError
    
    # User management
    async def create_user(self, user: User) -> bool:
        raise NotImplementedError
    
    async def get_user(self, user_id: str) -> Optional[User]:
        raise NotImplementedError
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        raise NotImplementedError
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        raise NotImplementedError
    
    async def update_user(self, user: User) -> bool:
        raise NotImplementedError
    
    async def delete_user(self, user_id: str) -> bool:
        raise NotImplementedError
    
    # Session management
    async def create_session(self, session: Session) -> bool:
        raise NotImplementedError
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        raise NotImplementedError
    
    async def update_session(self, session: Session) -> bool:
        raise NotImplementedError
    
    async def delete_session(self, session_id: str) -> bool:
        raise NotImplementedError
    
    async def cleanup_expired_sessions(self) -> int:
        raise NotImplementedError
    
    # Audit logging
    async def create_audit_log(self, audit_log: AuditLog) -> bool:
        raise NotImplementedError
    
    async def get_audit_logs(self, user_id: Optional[str] = None, 
                           action: Optional[str] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: int = 100) -> List[AuditLog]:
        raise NotImplementedError


class SQLiteProvider(DatabaseProvider):
    """SQLite database provider for development and testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("database_path", "rag_engine.db")
        self.connection = None
        
    async def connect(self):
        """Connect to SQLite database."""
        try:
            import aiosqlite
            self.connection = await aiosqlite.connect(self.db_path)
            await self._create_tables()
            logger.info(f"Connected to SQLite database: {self.db_path}")
        except ImportError:
            logger.error("aiosqlite package not installed. Run 'pip install aiosqlite'")
            raise
    
    async def disconnect(self):
        """Disconnect from SQLite database."""
        if self.connection:
            await self.connection.close()
            self.connection = None
    
    async def _create_tables(self):
        """Create database tables if they don't exist."""
        
        # Users table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                roles TEXT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Sessions table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        # Audit logs table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                log_id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                action TEXT NOT NULL,
                resource TEXT NOT NULL,
                details TEXT NOT NULL,
                ip_address TEXT NOT NULL,
                user_agent TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
        
        # Create indexes
        await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
        await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)")
        await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)")
        await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)")
        
        await self.connection.commit()
    
    async def create_user(self, user: User) -> bool:
        """Create a new user."""
        try:
            await self.connection.execute("""
                INSERT INTO users (user_id, username, email, password_hash, salt, roles, 
                                 is_active, created_at, last_login, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user.user_id, user.username, user.email, user.password_hash, user.salt,
                json.dumps(user.roles), user.is_active, user.created_at, user.last_login,
                json.dumps(user.metadata)
            ))
            await self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return False
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        try:
            cursor = await self.connection.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            )
            row = await cursor.fetchone()
            if row:
                return self._row_to_user(row)
            return None
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            cursor = await self.connection.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            )
            row = await cursor.fetchone()
            if row:
                return self._row_to_user(row)
            return None
        except Exception as e:
            logger.error(f"Failed to get user by username: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            cursor = await self.connection.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            )
            row = await cursor.fetchone()
            if row:
                return self._row_to_user(row)
            return None
        except Exception as e:
            logger.error(f"Failed to get user by email: {e}")
            return None
    
    def _row_to_user(self, row) -> User:
        """Convert database row to User object."""
        return User(
            user_id=row[0],
            username=row[1],
            email=row[2],
            password_hash=row[3],
            salt=row[4],
            roles=json.loads(row[5]),
            is_active=bool(row[6]),
            created_at=datetime.fromisoformat(row[7]) if row[7] else None,
            last_login=datetime.fromisoformat(row[8]) if row[8] else None,
            metadata=json.loads(row[9]) if row[9] else {}
        )
    
    async def update_user(self, user: User) -> bool:
        """Update an existing user."""
        try:
            await self.connection.execute("""
                UPDATE users SET username = ?, email = ?, password_hash = ?, salt = ?,
                               roles = ?, is_active = ?, last_login = ?, metadata = ?
                WHERE user_id = ?
            """, (
                user.username, user.email, user.password_hash, user.salt,
                json.dumps(user.roles), user.is_active, user.last_login,
                json.dumps(user.metadata), user.user_id
            ))
            await self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update user: {e}")
            return False
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        try:
            await self.connection.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            await self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete user: {e}")
            return False
    
    async def create_session(self, session: Session) -> bool:
        """Create a new session."""
        try:
            await self.connection.execute("""
                INSERT INTO sessions (session_id, user_id, created_at, last_accessed,
                                    expires_at, is_active, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id, session.user_id, session.created_at,
                session.last_accessed, session.expires_at, session.is_active,
                json.dumps(session.metadata)
            ))
            await self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        try:
            cursor = await self.connection.execute(
                "SELECT * FROM sessions WHERE session_id = ? AND is_active = TRUE", 
                (session_id,)
            )
            row = await cursor.fetchone()
            if row:
                return self._row_to_session(row)
            return None
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    def _row_to_session(self, row) -> Session:
        """Convert database row to Session object."""
        return Session(
            session_id=row[0],
            user_id=row[1],
            created_at=datetime.fromisoformat(row[2]),
            last_accessed=datetime.fromisoformat(row[3]),
            expires_at=datetime.fromisoformat(row[4]),
            is_active=bool(row[5]),
            metadata=json.loads(row[6]) if row[6] else {}
        )
    
    async def update_session(self, session: Session) -> bool:
        """Update an existing session."""
        try:
            await self.connection.execute("""
                UPDATE sessions SET last_accessed = ?, expires_at = ?, 
                                  is_active = ?, metadata = ?
                WHERE session_id = ?
            """, (
                session.last_accessed, session.expires_at, session.is_active,
                json.dumps(session.metadata), session.session_id
            ))
            await self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        try:
            await self.connection.execute(
                "UPDATE sessions SET is_active = FALSE WHERE session_id = ?", 
                (session_id,)
            )
            await self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        try:
            cursor = await self.connection.execute("""
                UPDATE sessions SET is_active = FALSE 
                WHERE expires_at < ? AND is_active = TRUE
            """, (datetime.now(),))
            await self.connection.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    async def create_audit_log(self, audit_log: AuditLog) -> bool:
        """Create a new audit log entry."""
        try:
            await self.connection.execute("""
                INSERT INTO audit_logs (log_id, user_id, session_id, action, resource,
                                      details, ip_address, user_agent, timestamp, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_log.log_id, audit_log.user_id, audit_log.session_id,
                audit_log.action, audit_log.resource, json.dumps(audit_log.details),
                audit_log.ip_address, audit_log.user_agent, audit_log.timestamp,
                audit_log.success
            ))
            await self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
            return False
    
    async def get_audit_logs(self, user_id: Optional[str] = None,
                           action: Optional[str] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: int = 100) -> List[AuditLog]:
        """Get audit logs with optional filtering."""
        try:
            query = "SELECT * FROM audit_logs WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if action:
                query += " AND action = ?"
                params.append(action)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = await self.connection.execute(query, params)
            rows = await cursor.fetchall()
            
            return [self._row_to_audit_log(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            return []
    
    def _row_to_audit_log(self, row) -> AuditLog:
        """Convert database row to AuditLog object."""
        return AuditLog(
            log_id=row[0],
            user_id=row[1],
            session_id=row[2],
            action=row[3],
            resource=row[4],
            details=json.loads(row[5]),
            ip_address=row[6],
            user_agent=row[7],
            timestamp=datetime.fromisoformat(row[8]),
            success=bool(row[9])
        )


class ProductionDatabaseManager:
    """Production database manager with multiple provider support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = self._create_provider()
        
    def _create_provider(self) -> DatabaseProvider:
        """Create database provider based on configuration."""
        provider_type = self.config.get("provider", "sqlite").lower()
        
        if provider_type == "sqlite":
            return SQLiteProvider(self.config)
        elif provider_type == "postgresql":
            # PostgreSQL provider would be implemented here
            raise NotImplementedError("PostgreSQL provider not yet implemented")
        elif provider_type == "mysql":
            # MySQL provider would be implemented here
            raise NotImplementedError("MySQL provider not yet implemented")
        else:
            raise ValueError(f"Unsupported database provider: {provider_type}")
    
    async def initialize(self):
        """Initialize the database connection and setup."""
        await self.provider.connect()
        logger.info("Production database initialized successfully")
    
    async def shutdown(self):
        """Shutdown the database connection."""
        await self.provider.disconnect()
        logger.info("Production database shutdown completed")
    
    # User management methods
    async def create_user(self, username: str, email: str, password: str, 
                         roles: List[str] = None) -> Optional[User]:
        """Create a new user with hashed password."""
        if roles is None:
            roles = ["user"]
        
        # Generate user ID and password hash
        user_id = secrets.token_urlsafe(16)
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            roles=roles
        )
        
        success = await self.provider.create_user(user)
        return user if success else None
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password."""
        user = await self.provider.get_user_by_username(username)
        if not user or not user.is_active:
            return None
        
        # Verify password
        password_hash = hashlib.pbkdf2_hmac(
            'sha256', password.encode(), user.salt.encode(), 100000
        ).hex()
        
        if password_hash == user.password_hash:
            # Update last login
            user.last_login = datetime.now()
            await self.provider.update_user(user)
            return user
        
        return None
    
    async def create_session(self, user_id: str, duration_hours: int = 24) -> Optional[Session]:
        """Create a new session for a user."""
        session_id = secrets.token_urlsafe(32)
        now = datetime.now()
        expires_at = now + timedelta(hours=duration_hours)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_accessed=now,
            expires_at=expires_at
        )
        
        success = await self.provider.create_session(session)
        return session if success else None
    
    async def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate and refresh a session."""
        session = await self.provider.get_session(session_id)
        if not session or session.is_expired():
            return None
        
        # Update last accessed time
        session.last_accessed = datetime.now()
        await self.provider.update_session(session)
        
        return session
    
    async def log_audit_event(self, user_id: Optional[str], session_id: Optional[str],
                            action: str, resource: str, details: Dict[str, Any],
                            ip_address: str, user_agent: str, success: bool = True):
        """Log an audit event."""
        log_id = secrets.token_urlsafe(16)
        
        audit_log = AuditLog(
            log_id=log_id,
            user_id=user_id,
            session_id=session_id,
            action=action,
            resource=resource,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now(),
            success=success
        )
        
        await self.provider.create_audit_log(audit_log)
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        return await self.provider.cleanup_expired_sessions()
    
    async def get_user_audit_logs(self, user_id: str, limit: int = 50) -> List[AuditLog]:
        """Get audit logs for a specific user."""
        return await self.provider.get_audit_logs(user_id=user_id, limit=limit) 