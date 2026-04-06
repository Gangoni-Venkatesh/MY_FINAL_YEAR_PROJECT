from django.db import models
from django.contrib.auth.models import User

class BlockchainDataset(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    csv_file = models.FileField(upload_to='datasets/')
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
