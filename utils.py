import csv


def load_csv_rows(csv_file_path, column_name):
    rows = []
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                rows.append(row[column_name])
            line_count += 1
    return rows
