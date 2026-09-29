from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("runs", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="LangFairMetricResult",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("cache_key", models.CharField(max_length=64)),
                ("result", models.JSONField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("run", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="langfair_results", to="runs.evaluationrun")),
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(fields=("run", "cache_key"), name="unique_langfair_result_cache"),
                ],
            },
        ),
    ]
