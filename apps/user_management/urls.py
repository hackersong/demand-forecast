from django.contrib.auth import views as auth_views
from django.urls import path, reverse_lazy

from . import views

app_name = 'user_management'

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('register/', views.register, name='register'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('logout/', views.logout_view, name='logout'),
    path('permissions/', views.permissions, name='permissions'),
    path('permissions/<int:user_id>/role/', views.set_user_role, name='set_role'),

    # Password change (needs the old password) and reset (needs a working mail
    # setup; the development default prints the mail to the console).
    path('password/change/', auth_views.PasswordChangeView.as_view(
        template_name='user_management/password_change.html',
        success_url=reverse_lazy('user_management:password_change_done'),
    ), name='password_change'),
    path('password/change/done/', auth_views.PasswordChangeDoneView.as_view(
        template_name='user_management/password_change_done.html',
    ), name='password_change_done'),
    path('password/reset/', auth_views.PasswordResetView.as_view(
        template_name='user_management/password_reset_form.html',
        email_template_name='user_management/password_reset_email.html',
        subject_template_name='user_management/password_reset_subject.txt',
        success_url=reverse_lazy('user_management:password_reset_done'),
    ), name='password_reset'),
    path('password/reset/done/', auth_views.PasswordResetDoneView.as_view(
        template_name='user_management/password_reset_done.html',
    ), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(
        template_name='user_management/password_reset_confirm.html',
        success_url=reverse_lazy('user_management:password_reset_complete'),
    ), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(
        template_name='user_management/password_reset_complete.html',
    ), name='password_reset_complete'),
]
