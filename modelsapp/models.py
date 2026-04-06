from django.db import models

class MLModel(models.Model):
    MODEL_CHOICES = [
        ('MLE', 'MLE Logistic Regression'),
        ('Bayes_HMC', 'Bayesian HMC Logistic Regression'),
        ('Bayes_Gibbs', 'Bayesian Gibbs Logistic Regression'),
    ]
    name = models.CharField(max_length=20, choices=MODEL_CHOICES)
    version = models.CharField(max_length=50, default='v1')
    pr_auc = models.FloatField()
    accuracy = models.FloatField()
    model_file = models.CharField(max_length=255)  # path to pickle in ml/models
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.get_name_display()} ({self.version})"
