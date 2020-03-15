from django import forms

class UserInputForm(forms.Form):
    anatom_site_general = forms.CharField(max_length=100)
    sex = forms.CharField(max_length=10)
    age_approx = forms.CharField(max_length=10)

