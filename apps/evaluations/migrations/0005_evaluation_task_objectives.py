from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("evaluations", "0004_evaluation_language")]

    operations = [
        migrations.AddField(
            model_name="evaluation",
            name="task_objectives",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
