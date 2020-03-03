# Write a Python program to convert a date of yyyy-mm-dd format to dd-mm-yyyy format.

my_date = "2020-01-30"
import datetime
new_date = datetime.datetime.strptime(my_date, "%Y-%m-%d").strftime("%d-%m-%Y")
print(new_date)