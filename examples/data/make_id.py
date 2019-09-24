import csv

with open("seg_b.csv", 'r') as f_in:
    with open("segmm_b.csv", 'w') as f_out:
        writer = csv.writer(f_out)
        rows = csv.reader(f_in)
        all = []
        newRow = []
        i = 0
        for row in rows:
            newRow = [i] + row
            all.append(newRow)
            i += 1
        writer.writerows(all)


