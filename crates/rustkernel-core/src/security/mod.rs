//! Security Infrastructure
//!
//! Enterprise security features for RustKernels:
//!
//! - **Authentication**: JWT/OAuth token validation
//! - **Authorization**: Role-based access control (RBAC)
//! - **Multi-tenancy**: Tenant isolation and resource quotas
//! - **Secrets**: Secure credential management
//!
//! # Feature Flags
//!
//! - `auth`: Enable authentication features
//! - `crypto`: Enable encryption features
//! - `tls`: Enable TLS support
//! - `multi-tenancy`: Enable multi-tenant isolation
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::security::{AuthConfig, SecurityContext};
//!
//! let config = AuthConfig::jwt("secret-key");
//! let ctx = SecurityContext::authenticate(&token, &config)?;
//!
//! // Check permissions
//! ctx.require_permission(KernelPermission::Execute)?;
//! ```

pub mod auth;
pub mod rbac;
pub mod secrets;
pub mod tenancy;

pub use auth::{AuthConfig, AuthProvider, AuthToken, TokenClaims};
pub use rbac::{KernelPermission, Permission, PermissionSet, Role, RoleBinding};
pub use secrets::{SecretRef, SecretStore, SecretValue};
pub use tenancy::{ResourceQuota, Tenant, TenantConfig, TenantId};

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Unified security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Authentication configuration
    pub auth: Option<AuthConfig>,
    /// Enable RBAC
    pub rbac_enabled: bool,
    /// Enable multi-tenancy
    pub multi_tenancy_enabled: bool,
    /// Default tenant ID for unauthenticated requests
    pub default_tenant: Option<TenantId>,
    /// Enable audit logging for security events
    pub audit_logging: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            auth: None,
            rbac_enabled: false,
            multi_tenancy_enabled: false,
            default_tenant: None,
            audit_logging: true,
        }
    }
}

impl SecurityConfig {
    /// Create a new security config
    pub fn new() -> Self {
        Self::default()
    }

    /// Development configuration (minimal security)
    pub fn development() -> Self {
        Self::default()
    }

    /// Production configuration
    pub fn production() -> Self {
        Self {
            auth: Some(AuthConfig::default()),
            rbac_enabled: true,
            multi_tenancy_enabled: true,
            default_tenant: None,
            audit_logging: true,
        }
    }

    /// Set auth configuration
    pub fn with_auth(mut self, config: AuthConfig) -> Self {
        self.auth = Some(config);
        self
    }

    /// Enable RBAC
    pub fn with_rbac(mut self, enabled: bool) -> Self {
        self.rbac_enabled = enabled;
        self
    }

    /// Enable multi-tenancy
    pub fn with_multi_tenancy(mut self, enabled: bool) -> Self {
        self.multi_tenancy_enabled = enabled;
        self
    }

    /// Set default tenant
    pub fn with_default_tenant(mut self, tenant: TenantId) -> Self {
        self.default_tenant = Some(tenant);
        self
    }
}

/// Security context for an authenticated request
#[derive(Debug, Clone)]
pub struct SecurityContext {
    /// Authenticated user ID
    pub user_id: Option<String>,
    /// Tenant ID
    pub tenant_id: Option<TenantId>,
    /// Roles assigned to the user
    pub roles: HashSet<Role>,
    /// Permissions derived from roles
    pub permissions: PermissionSet,
    /// Token claims (if JWT auth)
    pub claims: Option<TokenClaims>,
    /// Whether this is a system/service context
    pub is_system: bool,
}

impl SecurityContext {
    /// Create an anonymous context
    pub fn anonymous() -> Self {
        Self {
            user_id: None,
            tenant_id: None,
            roles: HashSet::new(),
            permissions: PermissionSet::empty(),
            claims: None,
            is_system: false,
        }
    }

    /// Create a system context with full permissions
    pub fn system() -> Self {
        Self {
            user_id: Some("system".to_string()),
            tenant_id: None,
            roles: {
                let mut roles = HashSet::new();
                roles.insert(Role::Admin);
                roles
            },
            permissions: PermissionSet::all(),
            claims: None,
            is_system: true,
        }
    }

    /// Create a user context
    pub fn user(user_id: impl Into<String>, tenant_id: Option<TenantId>) -> Self {
        Self {
            user_id: Some(user_id.into()),
            tenant_id,
            roles: HashSet::new(),
            permissions: PermissionSet::empty(),
            claims: None,
            is_system: false,
        }
    }

    /// Add a role
    pub fn with_role(mut self, role: Role) -> Self {
        let perms = role.permissions();
        self.roles.insert(role);
        // Update permissions based on role
        self.permissions = self.permissions.union(&perms);
        self
    }

    /// Add multiple roles
    pub fn with_roles(mut self, roles: impl IntoIterator<Item = Role>) -> Self {
        for role in roles {
            let perms = role.permissions();
            self.roles.insert(role);
            self.permissions = self.permissions.union(&perms);
        }
        self
    }

    /// Set claims
    pub fn with_claims(mut self, claims: TokenClaims) -> Self {
        self.claims = Some(claims);
        self
    }

    /// Check if user is authenticated
    pub fn is_authenticated(&self) -> bool {
        self.user_id.is_some()
    }

    /// Check if user has a permission
    pub fn has_permission(&self, permission: Permission) -> bool {
        self.is_system || self.permissions.contains(permission)
    }

    /// Check if user has a role
    pub fn has_role(&self, role: &Role) -> bool {
        self.is_system || self.roles.contains(role)
    }

    /// Require a permission, returning error if not granted
    pub fn require_permission(&self, permission: Permission) -> Result<(), SecurityError> {
        if self.has_permission(permission) {
            Ok(())
        } else {
            Err(SecurityError::PermissionDenied {
                permission: format!("{:?}", permission),
                user_id: self.user_id.clone(),
            })
        }
    }

    /// Require authentication
    pub fn require_authenticated(&self) -> Result<(), SecurityError> {
        if self.is_authenticated() {
            Ok(())
        } else {
            Err(SecurityError::Unauthenticated)
        }
    }

    /// Check if context can access a specific tenant
    pub fn can_access_tenant(&self, tenant_id: &TenantId) -> bool {
        self.is_system || self.tenant_id.as_ref() == Some(tenant_id) || self.has_role(&Role::Admin)
    }
}

impl Default for SecurityContext {
    fn default() -> Self {
        Self::anonymous()
    }
}

/// Security errors
#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    /// User is not authenticated
    #[error("Authentication required")]
    Unauthenticated,

    /// Token validation failed
    #[error("Invalid token: {reason}")]
    InvalidToken {
        /// The reason for token validation failure
        reason: String,
    },

    /// Token has expired
    #[error("Token expired")]
    TokenExpired,

    /// Permission denied
    #[error("Permission denied: {permission} for user {user_id:?}")]
    PermissionDenied {
        /// The permission that was denied
        permission: String,
        /// The user ID if available
        user_id: Option<String>,
    },

    /// Tenant access denied
    #[error("Tenant access denied: {tenant_id}")]
    TenantAccessDenied {
        /// The tenant ID that was denied access
        tenant_id: String,
    },

    /// Resource quota exceeded
    #[error("Resource quota exceeded: {resource}")]
    QuotaExceeded {
        /// The resource that exceeded quota
        resource: String,
    },

    /// Secret not found
    #[error("Secret not found: {name}")]
    SecretNotFound {
        /// The name of the secret that was not found
        name: String,
    },

    /// Encryption error
    #[error("Encryption error: {reason}")]
    EncryptionError {
        /// The reason for encryption failure
        reason: String,
    },

    /// Configuration error
    #[error("Security configuration error: {reason}")]
    ConfigError {
        /// The configuration error reason
        reason: String,
    },
}

impl From<SecurityError> for crate::error::KernelError {
    fn from(e: SecurityError) -> Self {
        match e {
            SecurityError::Unauthenticated
            | SecurityError::InvalidToken { .. }
            | SecurityError::TokenExpired
            | SecurityError::PermissionDenied { .. }
            | SecurityError::TenantAccessDenied { .. } => {
                crate::error::KernelError::Unauthorized(e.to_string())
            }
            SecurityError::QuotaExceeded { .. } => {
                crate::error::KernelError::ResourceExhausted(e.to_string())
            }
            _ => crate::error::KernelError::ConfigError(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anonymous_context() {
        let ctx = SecurityContext::anonymous();
        assert!(!ctx.is_authenticated());
        assert!(!ctx.is_system);
    }

    #[test]
    fn test_system_context() {
        let ctx = SecurityContext::system();
        assert!(ctx.is_authenticated());
        assert!(ctx.is_system);
        assert!(ctx.has_permission(Permission::KernelExecute));
        assert!(ctx.has_permission(Permission::KernelAdmin));
    }

    #[test]
    fn test_user_context() {
        let ctx = SecurityContext::user("user-123", Some(TenantId::new("tenant-456")))
            .with_role(Role::User);

        assert!(ctx.is_authenticated());
        assert!(!ctx.is_system);
        assert_eq!(ctx.user_id.as_deref(), Some("user-123"));
        assert!(ctx.has_permission(Permission::KernelExecute));
        assert!(!ctx.has_permission(Permission::KernelAdmin));
    }

    #[test]
    fn test_permission_check() {
        let ctx = SecurityContext::user("user-123", None).with_role(Role::User);

        assert!(ctx.require_permission(Permission::KernelExecute).is_ok());
        assert!(ctx.require_permission(Permission::KernelAdmin).is_err());
    }

    #[test]
    fn test_tenant_access() {
        let tenant = TenantId::new("tenant-123");
        let ctx = SecurityContext::user("user-123", Some(tenant.clone()));

        assert!(ctx.can_access_tenant(&tenant));
        assert!(!ctx.can_access_tenant(&TenantId::new("other-tenant")));
    }
}
