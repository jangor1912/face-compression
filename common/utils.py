import csv


def load_csv_rows(csv_file_path, column_name):
    rows = []
    with open(csv_file_path, mode='r', newline="\n") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            rows.append(row[column_name])
    return rows


def append_csv_row(csv_file_path, column_names, new_row):
    with open(csv_file_path, mode='a', newline="\n") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writerow(new_row)
