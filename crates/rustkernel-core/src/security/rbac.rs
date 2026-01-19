//! Role-Based Access Control (RBAC)
//!
//! Provides fine-grained access control for kernel operations.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Kernel-level permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    /// Execute kernels
    KernelExecute,
    /// Configure kernel parameters
    KernelConfigure,
    /// Monitor kernel metrics
    KernelMonitor,
    /// Administer kernels (register, unregister)
    KernelAdmin,
    /// Read kernel state
    StateRead,
    /// Write kernel state
    StateWrite,
    /// Access secrets
    SecretsRead,
    /// Manage secrets
    SecretsWrite,
    /// View tenant data
    TenantRead,
    /// Manage tenant settings
    TenantAdmin,
}

/// Kernel permission aliases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelPermission {
    /// Execute kernels
    Execute,
    /// Configure kernels
    Configure,
    /// Monitor kernels
    Monitor,
    /// Administer kernels
    Admin,
}

impl From<KernelPermission> for Permission {
    fn from(p: KernelPermission) -> Self {
        match p {
            KernelPermission::Execute => Permission::KernelExecute,
            KernelPermission::Configure => Permission::KernelConfigure,
            KernelPermission::Monitor => Permission::KernelMonitor,
            KernelPermission::Admin => Permission::KernelAdmin,
        }
    }
}

/// A set of permissions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PermissionSet {
    permissions: HashSet<Permission>,
}

impl PermissionSet {
    /// Create an empty permission set
    pub fn empty() -> Self {
        Self {
            permissions: HashSet::new(),
        }
    }

    /// Create a permission set with all permissions
    pub fn all() -> Self {
        let mut permissions = HashSet::new();
        permissions.insert(Permission::KernelExecute);
        permissions.insert(Permission::KernelConfigure);
        permissions.insert(Permission::KernelMonitor);
        permissions.insert(Permission::KernelAdmin);
        permissions.insert(Permission::StateRead);
        permissions.insert(Permission::StateWrite);
        permissions.insert(Permission::SecretsRead);
        permissions.insert(Permission::SecretsWrite);
        permissions.insert(Permission::TenantRead);
        permissions.insert(Permission::TenantAdmin);
        Self { permissions }
    }

    /// Check if the set contains a permission
    pub fn contains(&self, permission: Permission) -> bool {
        self.permissions.contains(&permission)
    }

    /// Add a permission
    pub fn add(&mut self, permission: Permission) {
        self.permissions.insert(permission);
    }

    /// Remove a permission
    pub fn remove(&mut self, permission: Permission) {
        self.permissions.remove(&permission);
    }

    /// Union with another permission set
    pub fn union(&self, other: &PermissionSet) -> PermissionSet {
        PermissionSet {
            permissions: self.permissions.union(&other.permissions).cloned().collect(),
        }
    }

    /// Intersection with another permission set
    pub fn intersection(&self, other: &PermissionSet) -> PermissionSet {
        PermissionSet {
            permissions: self
                .permissions
                .intersection(&other.permissions)
                .cloned()
                .collect(),
        }
    }

    /// Get all permissions
    pub fn iter(&self) -> impl Iterator<Item = &Permission> {
        self.permissions.iter()
    }
}

impl FromIterator<Permission> for PermissionSet {
    fn from_iter<I: IntoIterator<Item = Permission>>(iter: I) -> Self {
        Self {
            permissions: iter.into_iter().collect(),
        }
    }
}

/// Pre-defined roles
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Read-only access
    Viewer,
    /// Execute kernels only
    User,
    /// Execute and configure kernels
    Operator,
    /// Full access
    Admin,
    /// Custom role with specific permissions
    Custom(String),
}

impl Role {
    /// Get the permissions for this role
    pub fn permissions(&self) -> PermissionSet {
        match self {
            Role::Viewer => [Permission::KernelMonitor, Permission::StateRead]
                .into_iter()
                .collect(),
            Role::User => [
                Permission::KernelExecute,
                Permission::KernelMonitor,
                Permission::StateRead,
            ]
            .into_iter()
            .collect(),
            Role::Operator => [
                Permission::KernelExecute,
                Permission::KernelConfigure,
                Permission::KernelMonitor,
                Permission::StateRead,
                Permission::StateWrite,
            ]
            .into_iter()
            .collect(),
            Role::Admin => PermissionSet::all(),
            Role::Custom(_) => PermissionSet::empty(), // Custom roles need explicit permissions
        }
    }
}

/// Role binding - assigns a role to a user or group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleBinding {
    /// Binding name
    pub name: String,
    /// Role to bind
    pub role: Role,
    /// Subjects (users or groups)
    pub subjects: Vec<Subject>,
    /// Scope (namespace, tenant, etc.)
    pub scope: Option<String>,
}

impl RoleBinding {
    /// Create a new role binding
    pub fn new(name: impl Into<String>, role: Role) -> Self {
        Self {
            name: name.into(),
            role,
            subjects: Vec::new(),
            scope: None,
        }
    }

    /// Add a user subject
    pub fn for_user(mut self, user_id: impl Into<String>) -> Self {
        self.subjects.push(Subject::User(user_id.into()));
        self
    }

    /// Add a group subject
    pub fn for_group(mut self, group: impl Into<String>) -> Self {
        self.subjects.push(Subject::Group(group.into()));
        self
    }

    /// Set scope
    pub fn in_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }
}

/// Subject for role binding
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Subject {
    /// User subject
    User(String),
    /// Group subject
    Group(String),
    /// Service account
    ServiceAccount(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_set() {
        let mut perms = PermissionSet::empty();
        assert!(!perms.contains(Permission::KernelExecute));

        perms.add(Permission::KernelExecute);
        assert!(perms.contains(Permission::KernelExecute));

        perms.remove(Permission::KernelExecute);
        assert!(!perms.contains(Permission::KernelExecute));
    }

    #[test]
    fn test_all_permissions() {
        let perms = PermissionSet::all();
        assert!(perms.contains(Permission::KernelExecute));
        assert!(perms.contains(Permission::KernelAdmin));
        assert!(perms.contains(Permission::SecretsWrite));
    }

    #[test]
    fn test_role_permissions() {
        let viewer_perms = Role::Viewer.permissions();
        assert!(viewer_perms.contains(Permission::KernelMonitor));
        assert!(!viewer_perms.contains(Permission::KernelExecute));

        let user_perms = Role::User.permissions();
        assert!(user_perms.contains(Permission::KernelExecute));
        assert!(!user_perms.contains(Permission::KernelAdmin));

        let admin_perms = Role::Admin.permissions();
        assert!(admin_perms.contains(Permission::KernelAdmin));
    }

    #[test]
    fn test_permission_union() {
        let perms1: PermissionSet = [Permission::KernelExecute].into_iter().collect();
        let perms2: PermissionSet = [Permission::KernelMonitor].into_iter().collect();

        let union = perms1.union(&perms2);
        assert!(union.contains(Permission::KernelExecute));
        assert!(union.contains(Permission::KernelMonitor));
    }

    #[test]
    fn test_role_binding() {
        let binding = RoleBinding::new("admin-binding", Role::Admin)
            .for_user("user-123")
            .for_group("admins")
            .in_scope("tenant-456");

        assert_eq!(binding.role, Role::Admin);
        assert_eq!(binding.subjects.len(), 2);
        assert_eq!(binding.scope.as_deref(), Some("tenant-456"));
    }
}
