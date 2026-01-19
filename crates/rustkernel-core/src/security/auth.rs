//! Authentication
//!
//! JWT and OAuth authentication for kernel access.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication provider type
    pub provider: AuthProviderType,
    /// Token expiration time
    pub token_expiration: Duration,
    /// Allow anonymous access
    pub allow_anonymous: bool,
    /// Required claims for valid tokens
    pub required_claims: Vec<String>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            provider: AuthProviderType::Jwt {
                secret: String::new(),
                issuer: None,
                audience: None,
            },
            token_expiration: Duration::from_secs(3600),
            allow_anonymous: true,
            required_claims: Vec::new(),
        }
    }
}

impl AuthConfig {
    /// Create JWT authentication config
    pub fn jwt(secret: impl Into<String>) -> Self {
        Self {
            provider: AuthProviderType::Jwt {
                secret: secret.into(),
                issuer: None,
                audience: None,
            },
            ..Default::default()
        }
    }

    /// Set token expiration
    pub fn with_expiration(mut self, duration: Duration) -> Self {
        self.token_expiration = duration;
        self
    }

    /// Disable anonymous access
    pub fn require_auth(mut self) -> Self {
        self.allow_anonymous = false;
        self
    }

    /// Add required claim
    pub fn require_claim(mut self, claim: impl Into<String>) -> Self {
        self.required_claims.push(claim.into());
        self
    }
}

/// Authentication provider type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuthProviderType {
    /// JWT authentication
    Jwt {
        /// Secret key for HS256
        secret: String,
        /// Expected issuer
        issuer: Option<String>,
        /// Expected audience
        audience: Option<String>,
    },
    /// OAuth2/OIDC
    OAuth {
        /// Discovery URL
        discovery_url: String,
        /// Client ID
        client_id: String,
    },
    /// API Key
    ApiKey {
        /// Header name
        header: String,
    },
    /// No authentication (development only)
    None,
}

/// Authentication provider trait
pub trait AuthProvider: Send + Sync {
    /// Validate a token and extract claims
    fn validate(&self, token: &str) -> Result<TokenClaims, super::SecurityError>;

    /// Generate a new token for a user
    fn generate_token(&self, claims: &TokenClaims) -> Result<String, super::SecurityError>;
}

/// Authentication token
#[derive(Debug, Clone)]
pub struct AuthToken {
    /// Raw token string
    pub raw: String,
    /// Parsed claims
    pub claims: TokenClaims,
}

impl AuthToken {
    /// Create a new auth token
    pub fn new(raw: impl Into<String>, claims: TokenClaims) -> Self {
        Self {
            raw: raw.into(),
            claims,
        }
    }

    /// Get the user ID from claims
    pub fn user_id(&self) -> Option<&str> {
        self.claims.sub.as_deref()
    }

    /// Get the tenant ID from claims
    pub fn tenant_id(&self) -> Option<&str> {
        self.claims.tenant_id.as_deref()
    }

    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        if let Some(exp) = self.claims.exp {
            chrono::Utc::now().timestamp() as u64 > exp
        } else {
            false
        }
    }
}

/// JWT token claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClaims {
    /// Subject (user ID)
    pub sub: Option<String>,
    /// Issuer
    pub iss: Option<String>,
    /// Audience
    pub aud: Option<String>,
    /// Expiration time (Unix timestamp)
    pub exp: Option<u64>,
    /// Issued at (Unix timestamp)
    pub iat: Option<u64>,
    /// Not before (Unix timestamp)
    pub nbf: Option<u64>,
    /// JWT ID
    pub jti: Option<String>,
    /// Tenant ID (custom claim)
    pub tenant_id: Option<String>,
    /// Roles (custom claim)
    pub roles: Vec<String>,
    /// Permissions (custom claim)
    pub permissions: Vec<String>,
    /// Additional custom claims
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for TokenClaims {
    fn default() -> Self {
        Self {
            sub: None,
            iss: None,
            aud: None,
            exp: None,
            iat: Some(chrono::Utc::now().timestamp() as u64),
            nbf: None,
            jti: None,
            tenant_id: None,
            roles: Vec::new(),
            permissions: Vec::new(),
            extra: std::collections::HashMap::new(),
        }
    }
}

impl TokenClaims {
    /// Create new claims for a user
    pub fn for_user(user_id: impl Into<String>) -> Self {
        Self {
            sub: Some(user_id.into()),
            ..Default::default()
        }
    }

    /// Set expiration
    pub fn expires_in(mut self, duration: Duration) -> Self {
        let now = chrono::Utc::now().timestamp() as u64;
        self.exp = Some(now + duration.as_secs());
        self
    }

    /// Set tenant
    pub fn for_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Add role
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.roles.push(role.into());
        self
    }

    /// Add permission
    pub fn with_permission(mut self, permission: impl Into<String>) -> Self {
        self.permissions.push(permission.into());
        self
    }

    /// Add custom claim
    pub fn with_claim(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.extra.insert(key.into(), json_value);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_config() {
        let config = AuthConfig::jwt("my-secret")
            .with_expiration(Duration::from_secs(7200))
            .require_auth();

        assert!(!config.allow_anonymous);
        assert_eq!(config.token_expiration, Duration::from_secs(7200));
    }

    #[test]
    fn test_token_claims() {
        let claims = TokenClaims::for_user("user-123")
            .for_tenant("tenant-456")
            .with_role("admin")
            .expires_in(Duration::from_secs(3600));

        assert_eq!(claims.sub.as_deref(), Some("user-123"));
        assert_eq!(claims.tenant_id.as_deref(), Some("tenant-456"));
        assert!(claims.roles.contains(&"admin".to_string()));
        assert!(claims.exp.is_some());
    }

    #[test]
    fn test_auth_token() {
        let claims = TokenClaims::for_user("user-123");
        let token = AuthToken::new("raw-token-string", claims);

        assert_eq!(token.user_id(), Some("user-123"));
        assert!(!token.is_expired());
    }
}
