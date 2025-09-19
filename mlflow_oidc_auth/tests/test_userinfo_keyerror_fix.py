"""
Test cases for the KeyError: 'userinfo' bug fix.

This module contains tests that specifically verify the fix for the issue where
OIDC tokens lacking userinfo would cause KeyError exceptions in the authentication flow.

Issue: https://github.com/yurhett/mlflow-oidc-auth/issues/xxx
Original Error: KeyError: 'userinfo' in handle_user_and_group_management
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from mlflow_oidc_auth.auth import handle_user_and_group_management, process_oidc_callback


class TestUserinfoKeyErrorFix:
    """Test cases for the userinfo KeyError bug fix."""

    def test_handle_user_and_group_management_missing_userinfo_dict_key(self):
        """Test that missing userinfo dict key is handled gracefully."""
        # Token without userinfo key (original error scenario)
        token_without_userinfo = {"access_token": "token123", "id_token": "id_token123"}

        with patch("mlflow_oidc_auth.auth.app"):
            errors = handle_user_and_group_management(token_without_userinfo)

            assert len(errors) > 0
            assert any("userinfo" in error.lower() for error in errors)

    def test_handle_user_and_group_management_missing_userinfo_attribute(self):
        """Test that missing userinfo attribute is handled gracefully."""

        class TokenWithoutUserinfo:
            def __init__(self):
                self.access_token = "token123"
                # Missing userinfo attribute

        token = TokenWithoutUserinfo()

        with patch("mlflow_oidc_auth.auth.app"):
            errors = handle_user_and_group_management(token)

            assert len(errors) > 0
            assert any("userinfo" in error.lower() for error in errors)

    def test_handle_user_and_group_management_userinfo_not_dict(self):
        """Test that non-dict userinfo is handled gracefully."""
        token_bad_userinfo = {"access_token": "token123", "userinfo": "not_a_dict"}

        with patch("mlflow_oidc_auth.auth.app"):
            errors = handle_user_and_group_management(token_bad_userinfo)

            assert len(errors) > 0
            assert any("userinfo" in error and "dictionary" in error for error in errors)

    def test_handle_user_and_group_management_userinfo_none(self):
        """Test that None userinfo is handled gracefully."""
        token_none_userinfo = {"access_token": "token123", "userinfo": None}

        with patch("mlflow_oidc_auth.auth.app"):
            errors = handle_user_and_group_management(token_none_userinfo)

            assert len(errors) > 0
            assert any("userinfo" in error.lower() for error in errors)

    def test_process_oidc_callback_validates_userinfo_before_user_management(self):
        """Test that process_oidc_callback validates userinfo before calling user management."""
        mock_request = MagicMock()
        mock_request.args.get.side_effect = lambda k: "state_value" if k == "state" else None
        session = {"oauth_state": "state_value"}

        # Token without userinfo
        token_without_userinfo = {"access_token": "token123", "id_token": "id_token123"}

        with patch("mlflow_oidc_auth.auth.get_oauth_instance") as mock_oauth, patch(
            "mlflow_oidc_auth.auth.handle_token_validation", return_value=token_without_userinfo
        ), patch("mlflow_oidc_auth.auth.app"):

            mock_oauth.return_value.oidc = MagicMock()

            email, errors = process_oidc_callback(mock_request, session)

            # Should fail gracefully at userinfo validation, not in user management
            assert email is None
            assert len(errors) > 0
            assert any("userinfo" in error.lower() for error in errors)

    def test_process_oidc_callback_original_error_scenario(self):
        """Test the exact scenario from the original bug report."""
        mock_request = MagicMock()
        mock_request.args.get.side_effect = lambda k: {
            "state": "vXifYJenIKFjN_JFGS6rOg",
            "code": "15196095b1ff4e4e82e77d1b867cb7a9",
            "error": None,
            "error_description": None,
        }.get(k)

        session = {"oauth_state": "vXifYJenIKFjN_JFGS6rOg"}

        # Problematic token that caused the original KeyError
        problematic_token = {"access_token": "some_access_token", "id_token": "some_id_token", "token_type": "Bearer"}

        with patch("mlflow_oidc_auth.auth.get_oauth_instance") as mock_oauth, patch(
            "mlflow_oidc_auth.auth.handle_token_validation", return_value=problematic_token
        ), patch("mlflow_oidc_auth.auth.app"):

            mock_oauth.return_value.oidc = MagicMock()

            # This should not raise KeyError anymore
            email, errors = process_oidc_callback(mock_request, session)

            assert email is None
            assert len(errors) > 0
            assert any("userinfo" in error.lower() for error in errors)

    def test_handle_user_and_group_management_with_valid_userinfo_still_works(self):
        """Regression test to ensure valid userinfo scenarios still work."""
        token = {
            "userinfo": {"email": "user@example.com", "name": "Test User", "groups": ["users"]},
            "access_token": "token",
        }

        config = importlib.import_module("mlflow_oidc_auth.config").config
        config.OIDC_GROUP_DETECTION_PLUGIN = None
        config.OIDC_GROUPS_ATTRIBUTE = "groups"
        config.OIDC_ADMIN_GROUP_NAME = "admin"
        config.OIDC_GROUP_NAME = ["users"]

        with patch("mlflow_oidc_auth.auth.create_user") as mock_create, patch("mlflow_oidc_auth.auth.populate_groups") as mock_populate, patch(
            "mlflow_oidc_auth.auth.update_user"
        ) as mock_update, patch("mlflow_oidc_auth.auth.app"):

            errors = handle_user_and_group_management(token)

            # Should succeed
            assert errors == []
            mock_create.assert_called_once()
            mock_populate.assert_called_once()
            mock_update.assert_called_once()

    def test_process_oidc_callback_with_valid_userinfo_still_works(self):
        """Regression test to ensure valid callback scenarios still work."""
        mock_request = MagicMock()
        mock_request.args.get.side_effect = lambda k: "state_value" if k == "state" else None
        session = {"oauth_state": "state_value"}

        # Valid token with userinfo
        valid_token = {
            "access_token": "token123",
            "userinfo": {"email": "user@example.com", "name": "Test User", "groups": ["users"]},
        }

        with patch("mlflow_oidc_auth.auth.get_oauth_instance") as mock_oauth, patch(
            "mlflow_oidc_auth.auth.handle_token_validation", return_value=valid_token
        ), patch("mlflow_oidc_auth.auth.handle_user_and_group_management", return_value=[]), patch("mlflow_oidc_auth.auth.app"):

            mock_oauth.return_value.oidc = MagicMock()

            email, errors = process_oidc_callback(mock_request, session)

            # Should succeed
            assert email == "user@example.com"
            assert errors == []