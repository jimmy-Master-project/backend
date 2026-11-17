from django.db import models


class PersonaSet(models.Model):
    """A named, saved persona sample, reusable by any prompt population.

    Stores the conditions it was drawn with (filters on any MatrAIx
    dimension, size, seed) and the resulting persona ids, so a population
    generated from it uses exactly these people.
    """

    name = models.CharField(max_length=200)
    pool = models.CharField(max_length=200, blank=True, default="")
    # Keys are MatrAIx dimension names (data); values are the allowed values.
    filters = models.JSONField(default=dict, blank=True)
    requested_size = models.PositiveIntegerField()
    seed = models.BigIntegerField()
    persona_ids = models.JSONField(default=list, blank=True)
    # Full attribute snapshot of each persona at sampling time.
    personas = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self):
        return f"PersonaSet #{self.id} {self.name}"
