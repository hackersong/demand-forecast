"""Forms for registration and role assignment."""
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _

from .roles import ROLES


class RegisterForm(forms.Form):
    """Self-service registration.

    New accounts are created as viewers (read only); an admin upgrades them
    from the permissions page.
    """

    username = forms.CharField(
        label=_('用户名'), max_length=150,
        widget=forms.TextInput(attrs={'class': 'form-control', 'autofocus': True}),
    )
    email = forms.EmailField(
        label=_('邮箱'),
        widget=forms.EmailInput(attrs={'class': 'form-control'}),
    )
    password = forms.CharField(
        label=_('密码'), strip=False,
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
        help_text=_('至少 8 位，且不要与用户名过于相似。'),
    )
    password2 = forms.CharField(
        label=_('确认密码'), strip=False,
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
    )

    def clean_username(self):
        username = self.cleaned_data['username'].strip()
        if User.objects.filter(username__iexact=username).exists():
            raise ValidationError(_('该用户名已被占用，请换一个。'))
        return username

    def clean_email(self):
        email = self.cleaned_data['email'].strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise ValidationError(_('该邮箱已被注册，请直接登录或更换邮箱。'))
        return email

    def clean_password2(self):
        password = self.cleaned_data.get('password')
        password2 = self.cleaned_data.get('password2')
        if password and password2 and password != password2:
            raise ValidationError(_('两次输入的密码不一致。'))
        return password2

    def clean(self):
        cleaned = super().clean()
        password = cleaned.get('password')
        username = cleaned.get('username')
        if password:
            try:
                validate_password(
                    password,
                    user=User(username=username or '', email=cleaned.get('email') or ''),
                )
            except ValidationError as exc:
                self.add_error('password', exc)
        return cleaned

    def save(self):
        """Create the inactive-free user and return it (not yet logged in)."""
        user = User.objects.create_user(
            username=self.cleaned_data['username'],
            email=self.cleaned_data['email'],
            password=self.cleaned_data['password'],
        )
        from .roles import ROLE_VIEWER, set_role

        set_role(user, ROLE_VIEWER)
        return user


class RoleForm(forms.Form):
    """Role picker used on the permissions page."""

    role = forms.ChoiceField(
        label=_('角色'),
        choices=[(name, name) for name in ROLES],
        widget=forms.Select(attrs={'class': 'form-select form-select-sm'}),
    )
