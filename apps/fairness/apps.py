# pyright: reportAssignmentType=false
from django.apps import AppConfig


class FairnessConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.fairness"
