import csv
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
src = project_root / 'data' / 'raw' / 'pair_vicuna_jailbreaks.csv'
dst = project_root / 'data' / 'processed' / 'pair_vicuna_jailbreaks_annotated.csv'

rows = []
with src.open('r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) > 10:
            row = row[:9] + [','.join(row[9:])]
        rows.append(row)

for row in rows:
    if row and row[0] == '64':
        row[5] = 'None/Direct Request'
        row[6] = 'None'
        row[7] = 'Educational'
        row[8] = 'High'
        row[9] = 'Uses educational framing without a role assignment.'
        break

with dst.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f'Wrote {len(rows)-1} rows to {dst.name}')
print('Row 64:', next(r for r in rows if r and r[0] == '64'))
