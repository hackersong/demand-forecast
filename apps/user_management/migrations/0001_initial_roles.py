"""Create the three role groups and assign the users that already exist.

Roles are plain auth groups (see ``apps/user_management/roles.py``). Existing
staff and superusers become admins so they keep full access; everybody else
becomes a planner, which preserves the behaviour before roles existed.
"""
from django.db import migrations

ROLES = ('viewer', 'planner', 'admin')


def create_roles(apps, schema_editor):
    group_model = apps.get_model('auth', 'Group')
    for name in reversed(ROLES):
        group_model.objects.get_or_create(name=name)


def assign_existing_users(apps, schema_editor):
    user_model = apps.get_model('auth', 'User')
    group_model = apps.get_model('auth', 'Group')
    admin_group = group_model.objects.filter(name='admin').first()
    planner_group = group_model.objects.filter(name='planner').first()
    if admin_group is None or planner_group is None:
        return
    for user in user_model.objects.all():
        if user.is_superuser or user.is_staff:
            user.groups.add(admin_group)
        elif not user.groups.exists():
            user.groups.add(planner_group)


def remove_roles(apps, schema_editor):
    group_model = apps.get_model('auth', 'Group')
    group_model.objects.filter(name__in=ROLES).delete()


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.RunPython(create_roles, remove_roles),
        migrations.RunPython(assign_existing_users, migrations.RunPython.noop),
    ]
