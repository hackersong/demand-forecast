"""Account management: login, registration, roles, dashboard and passwords."""
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import gettext as _

from .forms import RegisterForm, RoleForm
from .roles import (
    ROLE_ADMIN,
    ROLE_PLANNER,
    ROLE_VIEWER,
    ROLES,
    WRITE_ROLES,
    ensure_roles,
    role_required,
    set_role,
    user_role,
)

# What every role is allowed to do, shown on the dashboard and permissions page.
ROLE_RIGHTS = {
    ROLE_VIEWER: _('查看数据洞察、分析结果与已有预测'),
    ROLE_PLANNER: _('查看权限 + 上传数据、训练模型、保存与调整预测'),
    ROLE_ADMIN: _('规划员权限 + 用户与角色管理'),
}


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        next_url = request.POST.get('next', '')

        if not username or not password:
            messages.error(request, _('请填写用户名和密码'))
            return render(request, 'user_management/login.html', {'next': next_url})

        user = authenticate(request, username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect(next_url or 'home')
            messages.error(request, _('该账户已被禁用'))
        else:
            messages.error(request, _('用户名或密码错误'))

        return render(request, 'user_management/login.html', {'next': next_url})

    return render(request, 'user_management/login.html',
                  {'next': request.GET.get('next', '')})


def logout_view(request):
    logout(request)
    return redirect('user_management:login')


def register(request):
    """Self-service registration. New accounts start as read-only viewers."""
    if request.user.is_authenticated:
        return redirect('user_management:dashboard')

    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(
                request,
                _('注册成功。当前角色为只读 viewer，如需上传数据或训练模型请联系管理员提升角色。'))
            return redirect('user_management:dashboard')
    else:
        form = RegisterForm()

    return render(request, 'user_management/register.html', {'form': form})


@login_required
def dashboard(request):
    role = user_role(request.user)
    context = {
        'role': role,
        'role_right': ROLE_RIGHTS.get(role, ''),
        'role_rights': ROLE_RIGHTS,
        'roles': ROLES,
        'can_write': role in WRITE_ROLES,
        'is_admin': role == ROLE_ADMIN,
    }
    if role == ROLE_ADMIN:
        ensure_roles()
        context['user_count'] = User.objects.count()
    return render(request, 'user_management/dashboard.html', context)


@login_required
@role_required(ROLE_ADMIN)
def permissions(request):
    """User list with a role picker. Admin only."""
    ensure_roles()
    rows = []
    for user in User.objects.all().order_by('-date_joined'):
        role = user_role(user)
        rows.append({
            'user': user,
            'role': role,
            'form': RoleForm(initial={'role': role}),
        })
    return render(request, 'user_management/permissions.html',
                  {'rows': rows, 'roles': ROLES,
                   'role_rights': [(role, ROLE_RIGHTS[role]) for role in ROLES]})


@login_required
@role_required(ROLE_ADMIN)
def set_user_role(request, user_id):
    """Change one user's role. Admin only, POST only."""
    if request.method != 'POST':
        return redirect('user_management:permissions')

    target = get_object_or_404(User, pk=user_id)
    form = RoleForm(request.POST)
    if not form.is_valid():
        messages.error(request, _('角色无效，请重新选择。'))
        return redirect('user_management:permissions')

    new_role = form.cleaned_data['role']
    # Never let an admin lock themselves out by downgrading their own account.
    if target == request.user and new_role != ROLE_ADMIN and not target.is_superuser:
        messages.error(request, _('不能取消自己的管理员角色，请由其他管理员操作。'))
        return redirect('user_management:permissions')

    set_role(target, new_role)
    messages.success(
        request,
        _('已将 %(name)s 的角色更新为 %(role)s。') % {'name': target.username, 'role': new_role})
    return redirect('user_management:permissions')
