"""Page level tests for every account flow.

Covers registration (including all validation failures), login, the dashboard,
role assignment, the role based write guard and both password flows.
"""
import re

from django.contrib.auth.models import User
from django.core import mail
from django.test import TestCase
from django.urls import reverse

from apps.user_management.roles import (
    ROLE_ADMIN,
    ROLE_PLANNER,
    ROLE_VIEWER,
    set_role,
    user_role,
)

REGISTER_URL = reverse('user_management:register')
LOGIN_URL = reverse('user_management:login')
DASHBOARD_URL = reverse('user_management:dashboard')
PERMISSIONS_URL = reverse('user_management:permissions')
UPLOAD_URL = reverse('data:upload')
TRAIN_URL = reverse('model:train')


def valid_payload(**overrides):
    payload = {
        'username': 'planner01',
        'email': 'planner01@example.com',
        'password': 'Strong-Pass-42',
        'password2': 'Strong-Pass-42',
    }
    payload.update(overrides)
    return payload


class RegisterViewTests(TestCase):
    def test_get_renders_form(self):
        response = self.client.get(REGISTER_URL)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'user_management/register.html')

    def test_register_creates_viewer_and_logs_in(self):
        response = self.client.post(REGISTER_URL, valid_payload())
        self.assertRedirects(response, DASHBOARD_URL)
        user = User.objects.get(username='planner01')
        self.assertEqual(user.email, 'planner01@example.com')
        self.assertTrue(user.check_password('Strong-Pass-42'))
        # New accounts are read only until an admin upgrades them.
        self.assertEqual(user_role(user), ROLE_VIEWER)
        # The user is signed in straight away.
        self.assertEqual(int(self.client.session['_auth_user_id']), user.pk)

    def test_duplicate_username_is_rejected(self):
        User.objects.create_user(username='planner01', password='Strong-Pass-42')
        response = self.client.post(REGISTER_URL, valid_payload(email='other@example.com'))
        self.assertEqual(response.status_code, 200)
        # The test client runs with the default language (English).
        self.assertFormError(
            response.context['form'], 'username',
            'This username is already taken, please choose another one.')
        self.assertEqual(User.objects.filter(username='planner01').count(), 1)

    def test_duplicate_email_is_rejected(self):
        User.objects.create_user(username='someone', email='planner01@example.com',
                                 password='Strong-Pass-42')
        response = self.client.post(REGISTER_URL, valid_payload())
        self.assertFormError(
            response.context['form'], 'email',
            'This email is already registered. Please log in or use another email.')

    def test_invalid_email_is_rejected(self):
        response = self.client.post(REGISTER_URL, valid_payload(email='not-an-email'))
        self.assertIn('email', response.context['form'].errors)

    def test_password_mismatch_is_rejected(self):
        response = self.client.post(REGISTER_URL, valid_payload(password2='Other-Pass-42'))
        self.assertFormError(
            response.context['form'], 'password2', 'The two passwords do not match.')

    def test_weak_password_is_rejected(self):
        # Too short and purely numeric, so both Django validators fire.
        response = self.client.post(REGISTER_URL, valid_payload(password='1234', password2='1234'))
        self.assertTrue(response.context['form'].errors.get('password'))
        self.assertFalse(User.objects.filter(username='planner01').exists())

    def test_logged_in_user_is_sent_to_the_dashboard(self):
        User.objects.create_user(username='existing', password='Strong-Pass-42')
        self.client.login(username='existing', password='Strong-Pass-42')
        self.assertRedirects(self.client.get(REGISTER_URL), DASHBOARD_URL)


class LoginLogoutTests(TestCase):
    def setUp(self):
        User.objects.create_user(username='viewer01', password='Strong-Pass-42')

    def test_login_page_renders(self):
        self.assertEqual(self.client.get(LOGIN_URL).status_code, 200)

    def test_login_and_logout_work(self):
        response = self.client.post(LOGIN_URL, {'username': 'viewer01',
                                                'password': 'Strong-Pass-42'})
        # Successful logins land on the home page (unchanged legacy behaviour).
        self.assertRedirects(response, '/')
        response = self.client.get(reverse('user_management:logout'))
        self.assertRedirects(response, LOGIN_URL)
        self.assertNotIn('_auth_user_id', self.client.session)

    def test_wrong_password_shows_error(self):
        response = self.client.post(LOGIN_URL,
                                    {'username': 'viewer01', 'password': 'wrong'})
        self.assertEqual(response.status_code, 200)
        # The test client runs with the default language (English).
        self.assertContains(response, 'Incorrect username or password')


class DashboardTests(TestCase):
    def test_requires_login(self):
        response = self.client.get(DASHBOARD_URL)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/user/login/', response['Location'])

    def test_shows_role_and_settings_links(self):
        user = User.objects.create_user(username='planner02', email='p2@example.com',
                                        password='Strong-Pass-42')
        set_role(user, ROLE_PLANNER)
        self.client.force_login(user)
        response = self.client.get(DASHBOARD_URL)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, ROLE_PLANNER)
        self.assertContains(response, 'p2@example.com')
        # Role management is only offered to admins.
        self.assertNotContains(response, PERMISSIONS_URL)


class RoleModelTests(TestCase):
    def test_user_without_groups_is_a_viewer(self):
        user = User.objects.create_user(username='plain', password='Strong-Pass-42')
        self.assertEqual(user_role(user), ROLE_VIEWER)
        self.assertIsNone(user_role(type('Anon', (), {'is_authenticated': False})()))

    def test_superuser_is_always_admin(self):
        user = User.objects.create_user(username='root', password='Strong-Pass-42',
                                        is_superuser=True)
        self.assertEqual(user_role(user), ROLE_ADMIN)

    def test_set_role_switches_group(self):
        user = User.objects.create_user(username='swap', password='Strong-Pass-42')
        set_role(user, ROLE_PLANNER)
        self.assertEqual(user_role(user), ROLE_PLANNER)
        set_role(user, ROLE_ADMIN)
        self.assertEqual(user_role(user), ROLE_ADMIN)
        self.assertEqual(user.groups.count(), 1)


class WriteGuardTests(TestCase):
    """Planner/admin may write, viewer may not."""

    def setUp(self):
        self.viewer = User.objects.create_user(username='v1', password='Strong-Pass-42')
        set_role(self.viewer, ROLE_VIEWER)
        self.planner = User.objects.create_user(username='p1', password='Strong-Pass-42')
        set_role(self.planner, ROLE_PLANNER)

    def test_viewer_cannot_open_write_pages(self):
        self.client.force_login(self.viewer)
        for url in (UPLOAD_URL, TRAIN_URL):
            response = self.client.get(url)
            self.assertRedirects(response, DASHBOARD_URL, msg_prefix=url)

    def test_planner_can_open_write_pages(self):
        self.client.force_login(self.planner)
        self.assertEqual(self.client.get(UPLOAD_URL).status_code, 200)
        self.assertEqual(self.client.get(TRAIN_URL).status_code, 200)

    def test_read_pages_stay_open_for_viewers(self):
        self.client.force_login(self.viewer)
        self.assertEqual(self.client.get(reverse('data:view_data')).status_code, 200)

    def test_admin_passes_everywhere(self):
        set_role(self.viewer, ROLE_ADMIN)
        self.client.force_login(self.viewer)
        self.assertEqual(self.client.get(UPLOAD_URL).status_code, 200)


class PermissionsPageTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_user(username='root', password='Strong-Pass-42',
                                              is_staff=True, is_superuser=True)
        self.planner = User.objects.create_user(username='p1', password='Strong-Pass-42')
        set_role(self.planner, ROLE_PLANNER)

    def test_admin_sees_user_list(self):
        self.client.force_login(self.admin)
        response = self.client.get(PERMISSIONS_URL)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'p1')

    def test_non_admin_is_rejected(self):
        self.client.force_login(self.planner)
        self.assertRedirects(self.client.get(PERMISSIONS_URL), DASHBOARD_URL)

    def test_admin_can_promote_a_viewer(self):
        viewer = User.objects.create_user(username='v9', password='Strong-Pass-42')
        set_role(viewer, ROLE_VIEWER)
        self.client.force_login(self.admin)
        response = self.client.post(
            reverse('user_management:set_role', args=[viewer.pk]), {'role': ROLE_PLANNER})
        self.assertRedirects(response, PERMISSIONS_URL)
        self.assertEqual(user_role(viewer), ROLE_PLANNER)

    def test_admin_cannot_downgrade_themselves(self):
        self.client.force_login(self.admin)
        self.client.post(reverse('user_management:set_role', args=[self.admin.pk]),
                         {'role': ROLE_VIEWER})
        self.assertEqual(user_role(self.admin), ROLE_ADMIN)


class PasswordChangeTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='u1', password='Old-Pass-42')

    def test_get_requires_login(self):
        response = self.client.get(reverse('user_management:password_change'))
        self.assertEqual(response.status_code, 302)

    def test_change_password_flow(self):
        self.client.force_login(self.user)
        response = self.client.post(reverse('user_management:password_change'), {
            'old_password': 'Old-Pass-42',
            'new_password1': 'New-Pass-42',
            'new_password2': 'New-Pass-42',
        })
        self.assertRedirects(
            response, reverse('user_management:password_change_done'))
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password('New-Pass-42'))
        # The session stays valid after the change.
        self.assertEqual(self.client.get(DASHBOARD_URL).status_code, 200)

    def test_wrong_old_password_is_rejected(self):
        self.client.force_login(self.user)
        response = self.client.post(reverse('user_management:password_change'), {
            'old_password': 'wrong',
            'new_password1': 'New-Pass-42',
            'new_password2': 'New-Pass-42',
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['form'].errors.get('old_password'))


class PasswordResetTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='u1', email='u1@example.com',
                                             password='Old-Pass-42')

    def test_reset_flow(self):
        response = self.client.post(reverse('user_management:password_reset'),
                                    {'email': 'u1@example.com'})
        self.assertRedirects(response, reverse('user_management:password_reset_done'))
        self.assertEqual(len(mail.outbox), 1)

        # Extract the reset link from the console email.
        body = mail.outbox[0].body
        path = re.search(r'http://testserver(/user/reset/\S+)', body).group(1)

        # Django answers the token link with a redirect that swaps the token
        # for `set-password/`, where the actual form lives.
        self.assertEqual(self.client.get(path).status_code, 302)
        final_url = path.rstrip('/').rsplit('/', 1)[0] + '/set-password/'
        confirm = self.client.get(final_url)
        self.assertEqual(confirm.status_code, 200)
        response = self.client.post(final_url, {
            'new_password1': 'New-Pass-42',
            'new_password2': 'New-Pass-42',
        })
        self.assertRedirects(response, reverse('user_management:password_reset_complete'))

        # The new password works.
        self.assertTrue(
            self.client.login(username='u1', password='New-Pass-42'))

    def test_unknown_email_does_not_leak(self):
        response = self.client.post(reverse('user_management:password_reset'),
                                    {'email': 'nobody@example.com'})
        self.assertRedirects(response, reverse('user_management:password_reset_done'))
        self.assertEqual(len(mail.outbox), 0)
