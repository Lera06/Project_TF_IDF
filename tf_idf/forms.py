from django import forms
from .models import FileUpload


class UploadForm(forms.ModelForm):

    class Meta:
        model = FileUpload
        fields = ('file', )
        widgets = {
            'file': forms.ClearableFileInput(attrs={'allow_multiple_selected': True})
        }

