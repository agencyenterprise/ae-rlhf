from ae_rlhf.config import settings


def test_settings_has_correct_env():
    """Verifies the tests are being run in the test environment."""
    assert settings.ENV == "test"
