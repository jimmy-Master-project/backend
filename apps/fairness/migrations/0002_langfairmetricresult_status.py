import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fairness", "0001_initial"),
    ]

    operations = [
        # Existing rows were only ever written after a successful computation.
        migrations.AddField(
            model_name="langfairmetricresult",
            name="status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("running", "Running"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                ],
                default="completed",
                max_length=16,
            ),
        ),
        migrations.AlterField(
            model_name="langfairmetricresult",
            name="status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("running", "Running"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                ],
                default="pending",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="langfairmetricresult",
            name="error",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AlterField(
            model_name="langfairmetricresult",
            name="result",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="langfairmetricresult",
            name="updated_at",
            field=models.DateTimeField(auto_now=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
    ]
