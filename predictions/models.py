from django.db import models
from django.contrib.auth.models import User
from modelsapp.models import MLModel

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    model = models.ForeignKey(MLModel, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    block_interval = models.FloatField()
    block_size = models.FloatField()
    fee_rate = models.FloatField()
    difficulty = models.FloatField()
    hash_rate = models.FloatField()
    mempool_tx_count = models.FloatField()

    is_delay = models.BooleanField()
    probability = models.FloatField()
    recommendation = models.CharField(max_length=255)

    def __str__(self):
        return f"Prediction #{self.id} by {self.user}"
