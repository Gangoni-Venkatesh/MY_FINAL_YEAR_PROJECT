from django import forms
from .models import BlockchainDataset

class BlockchainDatasetForm(forms.ModelForm):
    class Meta:
        model = BlockchainDataset
        fields = ['name', 'description', 'csv_file']
