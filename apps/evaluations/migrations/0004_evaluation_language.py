from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("evaluations", "0003_evaluation_recommendation_items")]

    operations = [
        migrations.AddField(
            model_name="evaluation",
            name="language",
            field=models.CharField(
                choices=[
                    ("en", "English"),
                    ("zh-TW", "Traditional Chinese"),
                    ("auto", "Auto detect"),
                ],
                default="en",
                max_length=8,
            ),
        ),
    ]
