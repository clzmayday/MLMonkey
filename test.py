import csv, os

# Read CSV files containing the all defect information
csv_head = []
csv_data = []
with open ("./Calculator.csv", "r") as csvFile:
    csvReader = [i for i in csv.reader(csvFile)]
    csv_head = csvReader[0]
    csv_data = csvReader[1:]
    csvFile.close()


print()