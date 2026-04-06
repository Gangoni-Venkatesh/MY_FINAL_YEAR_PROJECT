from django import forms
from modelsapp.models import MLModel

class PredictionInputForm(forms.Form):
    model = forms.ModelChoiceField(
        queryset=MLModel.objects.filter(is_active=True),
        label='Model'
    )
    block_interval = forms.FloatField(label='Block interval (seconds)')
    block_size = forms.FloatField(label='Block size (MB)')
    fee_rate = forms.FloatField(label='Fee rate (sat/vByte)')
    difficulty = forms.FloatField(label='Difficulty')
    hash_rate = forms.FloatField(label='Hash rate (EH/s)')
    mempool_tx_count = forms.FloatField(label='Mempool transaction count')
