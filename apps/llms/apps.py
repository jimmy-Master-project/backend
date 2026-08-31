# pyright: reportAssignmentType=false
from django.apps import AppConfig


class LlmsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.llms"
    label = "llms"
