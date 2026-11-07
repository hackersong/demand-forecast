"""Roles for the demand forecasting system.

Roles are plain Django auth groups, so the whole permission machinery stays in
Django (admin site, `user.groups`, future per-object permissions). Three roles
are created by a migration:

* ``viewer``  - read only: dashboards, insight reports, existing forecasts
* ``planner`` - viewer plus every write operation: data upload, training,
                saving or adjusting forecasts
* ``admin``   - planner plus user and role management

Users without an explicit role fall back to ``viewer``. Superusers are always
treated as ``admin``, which keeps `createsuperuser` working even before roles
are assigned.
"""
from functools import wraps

from django.contrib import messages
from django.contrib.auth.models import Group
from django.shortcuts import redirect
from django.utils.translation import gettext as _

ROLE_VIEWER = 'viewer'
ROLE_PLANNER = 'planner'
ROLE_ADMIN = 'admin'

#: Ordered from the most to the least privileged.
ROLES = (ROLE_ADMIN, ROLE_PLANNER, ROLE_VIEWER)

#: Roles allowed to change data, train models and edit forecasts.
WRITE_ROLES = (ROLE_ADMIN, ROLE_PLANNER)


def ensure_roles():
    """Create the three role groups. Idempotent, safe to call at any time."""
    return [Group.objects.get_or_create(name=name)[0] for name in reversed(ROLES)]


def user_role(user):
    """Return the role name of ``user`` (or None for anonymous visitors)."""
    if user is None or not user.is_authenticated:
        return None
    if user.is_superuser:
        return ROLE_ADMIN
    names = set(user.groups.values_list('name', flat=True))
    for role in ROLES:
        if role in names:
            return role
    # Users created before the roles existed keep working as viewers; an admin
    # upgrades them from the permissions page.
    return ROLE_VIEWER


def set_role(user, role):
    """Assign exactly one role group to ``user``."""
    if role not in ROLES:
        raise ValueError('unknown role: %r' % (role,))
    groups = [Group.objects.get_or_create(name=name)[0] for name in ROLES]
    keep = next(g for g in groups if g.name == role)
    user.groups.set([keep])


def role_required(*roles):
    """Reject users whose role is not in ``roles`` instead of executing it."""
    allowed = tuple(roles)

    def decorator(view_func):
        @wraps(view_func)
        def wrapped(request, *args, **kwargs):
            if user_role(request.user) in allowed:
                return view_func(request, *args, **kwargs)
            messages.error(request, _('当前角色没有该操作的权限，请联系管理员调整角色。'))
            return redirect('user_management:dashboard')

        return wrapped

    return decorator
