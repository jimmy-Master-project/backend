from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("llms", "0001_initial")]

    operations = [
        migrations.AddField(
            model_name="llmconfig",
            name="api_key",
            field=models.CharField(blank=True, default="", max_length=512),
        ),
    ]
